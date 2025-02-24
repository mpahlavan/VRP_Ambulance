from marpdan.dep import ORTOOLS_ENABLED, pywrapcp, routing_enums_pb2
from marpdan.dep import tqdm
from multiprocessing import Pool

def _solve_cp(nodes, veh_count, veh_capa, veh_speed, spoilage_penalty):
    """Solve single PVRP instance using OR-Tools"""
    manager = pywrapcp.RoutingIndexManager(nodes.size(0), veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)

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
    
    routing.AddDimension(
        time_callback_idx,
        0,  # no slack
        max_time,  # maximum time
        True,  # start cumul to zero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return 1 if from_node != 0 else 0

    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx,
        0,  # no slack
        [veh_capa] * veh_count,
        True,  # start cumul to zero
        "Capacity"
    )

    # Calculate depot return times
    depot_pos = nodes[0, :2]
    depot_return_times = []
    for i in range(nodes.size(0)):
        node_pos = nodes[i, :2]
        dist_to_depot = (node_pos - depot_pos).pow(2).sum().pow(0.5)
        depot_return_times.append(int(dist_to_depot / veh_speed))

    # Add spoilage constraints
    for node in range(1, nodes.size(0)):
        index = manager.NodeToIndex(node)
        spoilage_time = int(nodes[node, 3].item())
        latest_pickup = spoilage_time - depot_return_times[node]
        
        if latest_pickup <= 0:
            continue
            
        time_dimension.CumulVar(index).SetRange(0, max(1, latest_pickup))
        time_dimension.SetCumulVarSoftUpperBound(
            index, 
            latest_pickup,
            spoilage_penalty
        )

    # Solver settings
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30  # Reduced timeout
    search_parameters.solution_limit = 1

    try:
        solution = routing.SolveWithParameters(search_parameters)
        if not solution:
            return []

        routes = []
        for vehicle_id in range(veh_count):
            route = []
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx != 0:
                    route.append(node_idx)
                index = solution.Value(routing.NextVar(index))
            if route:
                routes.append(route)
        return routes
    except Exception:
        return []

def ort_solve(data, spoilage_penalty=10):
    """Solve PVRP instances using OR-Tools"""
    try:
        from concurrent.futures import ProcessPoolExecutor, TimeoutError
        with ProcessPoolExecutor() as executor:
            futures = []
            for nodes in data.nodes_gen():
                future = executor.submit(
                    _solve_cp,
                    nodes, data.veh_count, data.veh_capa, 
                    data.veh_speed, spoilage_penalty
                )
                futures.append(future)
            
            routes = []
            with tqdm(total=len(futures), desc="Solving with OR-Tools") as pbar:
                for future in futures:
                    try:
                        route = future.result(timeout=60)
                        routes.append(route)
                        pbar.update(1)
                    except TimeoutError:
                        routes.append([])
                        pbar.update(1)
                    except Exception:
                        routes.append([])
                        pbar.update(1)
            
            if routes and routes[0]:
                print("\nExample solution:")
                print_solution(routes[0], next(data.nodes_gen()), data.veh_speed)
            
            return routes
    except KeyboardInterrupt:
        print("\nSolver interrupted. Proceeding with training...")
        return None



# def _solve_cp(nodes, veh_count, veh_capa, veh_speed, spoilage_penalty):
#     """Solve single PVRP instance using OR-Tools"""
    
#     # Create routing manager
#     manager = pywrapcp.RoutingIndexManager(nodes.size(0), veh_count, 0)
#     routing = pywrapcp.RoutingModel(manager)

#     # Distance callback
#     def distance_callback(from_index, to_index):
#         from_node = manager.IndexToNode(from_index)
#         to_node = manager.IndexToNode(to_index)
#         return int(nodes[from_node, :2].sub(nodes[to_node, :2]).pow(2).sum().pow(0.5))

#     dist_callback_idx = routing.RegisterTransitCallback(distance_callback)
#     routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_idx)

#     # Time callback including travel times
#     def time_callback(from_index, to_index):
#         from_node = manager.IndexToNode(from_index)
#         to_node = manager.IndexToNode(to_index)
#         # Travel time between nodes
#         return int(distance_callback(from_index, to_index) / veh_speed)

#     time_callback_idx = routing.RegisterTransitCallback(time_callback)

#     # Add Time dimension
#     max_time = int(nodes[:, 3].max().item())
#     routing.AddDimension(
#         time_callback_idx,
#         0,  # no slack
#         max_time,  # maximum time (spoilage deadline)
#         True,  # start cumul to zero
#         "Time"
#     )
#     time_dimension = routing.GetDimensionOrDie("Time")

#     # Add capacity constraints (unit demands)
#     def demand_callback(from_index):
#         from_node = manager.IndexToNode(from_index)
#         return 1 if from_node != 0 else 0  # Unit demand except depot

#     demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
#     routing.AddDimensionWithVehicleCapacity(
#         demand_callback_idx,
#         0,  # no slack
#         [veh_capa] * veh_count,  # vehicle capacities
#         True,  # start cumul to zero
#         "Capacity"
#     )
#     # Calculate depot return times for each node first
#     depot_pos = nodes[0, :2]
#     depot_return_times = []
#     for i in range(nodes.size(0)):
#         node_pos = nodes[i, :2]
#         dist_to_depot = (node_pos - depot_pos).pow(2).sum().pow(0.5)
#         return_time = int(dist_to_depot / veh_speed)
#         depot_return_times.append(return_time)

#     # Add spoilage time constraints with depot return consideration
#     for node in range(1, nodes.size(0)):  # Skip depot
#         index = manager.NodeToIndex(node)
#         spoilage_time = int(nodes[node, 3].item())
#         # Latest pickup time = spoilage time - time to return to depot
#         latest_pickup = spoilage_time - depot_return_times[node]
        

#         if latest_pickup <= 0:
#             continue  # Skip nodes that can't be reached in time
#         # Add time windows for spoilage
#         time_dimension.CumulVar(index).SetRange(0,  max(1, latest_pickup))
        
#         # Add penalty for approaching spoilage time
#         time_dimension.SetCumulVarSoftUpperBound(
#             index, 
#             latest_pickup,  # Use latest pickup time for penalty too
#             spoilage_penalty
#         )
#     # Solver settings
#     # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#     # search_parameters.first_solution_strategy = (
#     #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
#     # )
#     # search_parameters.local_search_metaheuristic = (
#     #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
#     # )
#     # search_parameters.time_limit.FromSeconds(30)
    
    
#     search_parameters = pywrapcp.DefaultRoutingSearchParameters()
#     # Use SAVINGS for better initial clustering
#     search_parameters.first_solution_strategy = (
#        routing_enums_pb2.FirstSolutionStrategy.SAVINGS
#     )
#     # Use TABU_SEARCH to escape local optima
#     search_parameters.local_search_metaheuristic = (
#     routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
#     )
#     search_parameters.time_limit.FromSeconds(6000)
#     search_parameters.solution_limit = 100

#     # Solve
#     solution = routing.SolveWithParameters(search_parameters)
#     if not solution:
#         return []


#     # Extract routes
#     routes = []
#     for vehicle_id in range(veh_count):
#         route = []
#         index = routing.Start(vehicle_id)
#         while not routing.IsEnd(index):
#             node_idx = manager.IndexToNode(index)
#             if node_idx != 0:  # Don't include depot in middle of route
#                 route.append(node_idx)
#             index = solution.Value(routing.NextVar(index))
#         if route:  # Only add non-empty routes
#             routes.append(route)

#     return routes




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

# def ort_solve(data, spoilage_penalty=10):
#     """Solve PVRP instances using OR-Tools"""
#     with Pool() as p:
#         with tqdm(desc="Calling ORTools", total=data.batch_size) as pbar:
#             results = [
#                 p.apply_async(
#                     _solve_cp,
#                     (nodes, data.veh_count, data.veh_capa, data.veh_speed, spoilage_penalty),
#                     callback=lambda _: pbar.update()
#                 ) for nodes in data.nodes_gen()
#             ]
#             routes = [res.get() for res in results]
            
#             # Print first solution for debugging
#             if routes and routes[0]:
#                 print("\nExample solution:")
#                 print_solution(routes[0], next(data.nodes_gen()), data.veh_speed)
                
#     return routes