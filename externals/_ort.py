from marpdan.dep import ORTOOLS_ENABLED, pywrapcp, routing_enums_pb2
from marpdan.dep import tqdm
from multiprocessing import Pool
from marpdan.utils import parse_args

def _solve_cp(nodes, veh_count, veh_capa, veh_speed,
              spoilage_penalty, unserved_penalty, dist_penalty_coef):
    """Solve single PVRP instance using OR-Tools.

    Coverage is a soft constraint: every patient carries a drop penalty
    equivalent to the RL environment's unserved-patient penalty, so the
    solver may trade coverage against lateness on the same terms.
    """

    manager = pywrapcp.RoutingIndexManager(nodes.size(0), veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    # --- objective is in distance units; convert RL reward units to match ---
    # RL cost:  alpha2 * distance  +  alpha1 * late  +  alpha4 * unserved
    # OR-Tools: 1.0   * distance  +  P_late * late   +  P_drop * unserved
    # so   P = alpha / alpha2
    unit = float(dist_penalty_coef) if dist_penalty_coef > 0 else 1.0
    late_penalty_ort = int(round(spoilage_penalty / unit))
    drop_penalty_ort = int(round(unserved_penalty / unit))

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(nodes[from_node, :2].sub(nodes[to_node, :2]).pow(2).sum().pow(0.5))

    dist_callback_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_idx)

    def time_callback(from_index, to_index):
        return int(distance_callback(from_index, to_index) / veh_speed)

    time_callback_idx = routing.RegisterTransitCallback(time_callback)

    max_time = int(nodes[:, 3].max().item())
    routing.AddDimension(time_callback_idx, 0, max_time, True, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    def demand_callback(from_index):
        return 1 if manager.IndexToNode(from_index) != 0 else 0

    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx, 0, [veh_capa] * veh_count, True, "Capacity"
    )

    depot_pos = nodes[0, :2]
    depot_return_times = [
        int((nodes[i, :2] - depot_pos).pow(2).sum().pow(0.5) / veh_speed)
        for i in range(nodes.size(0))
    ]

    for node in range(1, nodes.size(0)):
        index = manager.NodeToIndex(node)
        spoilage_time = int(nodes[node, 3].item())
        latest_pickup = spoilage_time - depot_return_times[node]

        if latest_pickup <= 0:
            # Structurally infeasible: excluded from the RL feasibility mask too,
            # so it is dropped at zero cost and does not distort the objective.
            routing.AddDisjunction([index], 0)
            continue

        # Coverage is optional, priced at the RL unserved-patient penalty.
        routing.AddDisjunction([index], drop_penalty_ort)

        # Pickup deadline is soft, priced at the RL lateness penalty.
        time_dimension.CumulVar(index).SetRange(0, max_time)
        if late_penalty_ort > 0:
            time_dimension.SetCumulVarSoftUpperBound(
                index, latest_pickup, late_penalty_ort
            )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    )
    search_parameters.time_limit.FromSeconds(6000)
    search_parameters.solution_limit = 100

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return [[] for _ in range(veh_count)]

    routes = []
    for vehicle_id in range(veh_count):
        route = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx != 0:
                route.append(node_idx)
            index = solution.Value(routing.NextVar(index))
        routes.append(route)          # ← بدون شرط if route

    return routes




def print_solution(routes, nodes, veh_speed):
    """Print readable solution"""
    total_distance = 0
    total_time = 0
    
    for i, route in enumerate(routes):
        if not route:
            continue
            
        print(f"\nRoute {i}:")
        time = 0
        distance = 0
        current_pos = nodes[0, :2]  # Start at depot
        
        print(f"Depot -> ", end='')
        for node in route:
            # Calculate metrics to next node
            next_pos = nodes[node, :2]
            dist = (next_pos - current_pos).pow(2).sum().pow(0.5)
            travel_time = dist / veh_speed
            
            # Update cumulative metrics
            distance += dist
            time += travel_time
            
            # Print node info
            print(f"{node}(t={time:.1f},d={distance:.1f}) -> ", end='')
            
            current_pos = next_pos
            
        # Return to depot
        dist = (nodes[0, :2] - current_pos).pow(2).sum().pow(0.5)
        distance += dist
        time += dist / veh_speed
        print(f"Depot(t={time:.1f},d={distance:.1f})")
        
        total_distance += distance
        total_time += time
        
    print(f"\nTotal distance: {total_distance:.1f}")
    print(f"Total time: {total_time:.1f}")

def ort_solve(data):
    args = parse_args()
    with Pool() as p:
        with tqdm(desc="Calling ORTools", total=data.batch_size) as pbar:
            results = [
                p.apply_async(
                    _solve_cp,
                    (nodes, data.veh_count, data.veh_capa, data.veh_speed,
                     args.spoilage_penalty, args.unserved_penalty,
                     args.dist_penalty_coef),
                    callback=lambda _: pbar.update()
                ) for nodes in data.nodes_gen()
            ]
            routes = [res.get() for res in results]
    return routes