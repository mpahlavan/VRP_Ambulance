from marpdan.dep import ORTOOLS_ENABLED, pywrapcp, routing_enums_pb2
from marpdan.dep import tqdm
from multiprocessing import Pool

def _solve_cp(nodes, veh_count, veh_capa, veh_speed, spoilage_penalty=2.0, pickup_bonus=1.0, dist_penalty=1.0):
    """
    Solve single PVRP instance using OR-Tools with enhanced penalty structure
    """
    # Create routing manager
    manager = pywrapcp.RoutingIndexManager(nodes.size(0), veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback with enhanced penalty
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        dist = nodes[from_node, :2].sub(nodes[to_node, :2]).pow(2).sum().pow(0.5).item()
        return int(dist * dist_penalty * 100)  # Scale for integer conversion and apply penalty

    dist_callback_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_idx)

    # Time callback including travel times
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        dist = nodes[from_node, :2].sub(nodes[to_node, :2]).pow(2).sum().pow(0.5).item()
        return int((dist / veh_speed) * 100)  # Scale for precision

    time_callback_idx = routing.RegisterTransitCallback(time_callback)

    # Add Time dimension
    max_time = int(nodes[:, 3].max().item() * 100)  # Scale for precision
    routing.AddDimension(
        time_callback_idx,
        0,  # no slack
        max_time,  # maximum time (spoilage deadline)
        True,  # start cumul to zero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Add capacity constraints (unit demands)
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return 1 if from_node != 0 else 0  # Unit demand except depot

    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx,
        0,  # no slack
        [veh_capa] * veh_count,  # vehicle capacities
        True,  # start cumul to zero
        "Capacity"
    )
    
    # Calculate depot return times for each node
    depot_pos = nodes[0, :2]
    depot_return_times = []
    for i in range(nodes.size(0)):
        node_pos = nodes[i, :2]
        dist_to_depot = (node_pos - depot_pos).pow(2).sum().pow(0.5).item()
        return_time = int((dist_to_depot / veh_speed) * 100)  # Scale for precision
        depot_return_times.append(return_time)

    # Add spoilage time constraints with depot return consideration
    for node in range(1, nodes.size(0)):  # Skip depot
        index = manager.NodeToIndex(node)
        spoilage_time = int(nodes[node, 3].item() * 100)  # Scale for precision
        
        # Latest pickup time = spoilage time - time to return to depot
        latest_pickup = spoilage_time - depot_return_times[node]
        
        if latest_pickup <= 0:
            continue  # Skip nodes that can't be reached in time
        
        # Add time windows for spoilage
        time_dimension.CumulVar(index).SetRange(0, max(1, latest_pickup))
        
        # Add penalty for approaching spoilage time (late pickups)
        time_dimension.SetCumulVarSoftUpperBound(
            index, 
            int(latest_pickup * 0.9),  # Set penalty to start at 90% of time window
            int(spoilage_penalty * 100)  # Scale penalty for integer conversion
        )
        
        # Add bonus for early pickups
        if pickup_bonus > 0:
            early_threshold = int(latest_pickup * 0.5)  # Bonus for pickups in first half of window
            time_dimension.SetCumulVarSoftLowerBound(
                index,
                early_threshold,
                int(pickup_bonus * 100)  # Scale bonus for integer conversion
            )
    
    # Add pickup bonus for each node visited
    for node in range(1, nodes.size(0)):  # Skip depot
        index = manager.NodeToIndex(node)
        routing.AddDisjunction([index], 0, int(pickup_bonus * 200))  # Doubled bonus for priority
    
    # Solver settings
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Use SAVINGS for better initial clustering
    search_parameters.first_solution_strategy = (
       routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    )
    # Use TABU_SEARCH to escape local optima
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    )
    search_parameters.time_limit.FromSeconds(60)  # Limit search time
    search_parameters.solution_limit = 100

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return []

    # Extract routes
    routes = []
    for vehicle_id in range(veh_count):
        route = []
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx != 0:  # Don't include depot in middle of route
                route.append(node_idx)
            index = solution.Value(routing.NextVar(index))
        if route:  # Only add non-empty routes
            routes.append(route)

    return routes

def print_solution(routes, nodes, veh_speed):
    """Print readable solution with enhanced metrics"""
    total_distance = 0
    total_time = 0
    total_nodes_served = 0
    on_time_pickups = 0
    late_pickups = 0
    
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
            dist = (next_pos - current_pos).pow(2).sum().pow(0.5).item()
            travel_time = dist / veh_speed
            
            # Update cumulative metrics
            distance += dist
            time += travel_time
            total_nodes_served += 1
            
            # Check if pickup is on time
            node_spoilage = nodes[node, 3].item()
            dist_to_depot = (next_pos - nodes[0, :2]).pow(2).sum().pow(0.5).item()
            time_to_depot = dist_to_depot / veh_speed
            
            # If current time + time to depot > spoilage time, it's a late pickup
            if time + time_to_depot > node_spoilage:
                late_status = "LATE"
                late_pickups += 1
            else:
                late_status = "On-time"
                on_time_pickups += 1
            
            # Print node info with status
            print(f"{node}(t={time:.1f},d={distance:.1f}) -> ", end='')
            
            current_pos = next_pos
            
        # Return to depot
        dist = (nodes[0, :2] - current_pos).pow(2).sum().pow(0.5).item()
        distance += dist
        time += dist / veh_speed
        print(f"Depot(t={time:.1f},d={distance:.1f})")
        
        total_distance += distance
        total_time += time
        
    print(f"\nTotal distance: {total_distance:.1f}")
    print(f"Total time: {total_time:.1f}")

def ort_solve(data, args=None):
    """
    Solve PVRP instances using OR-Tools with parameters from args
    
    Args:
        data: PVRP Dataset
        args: Command line arguments (optional)
    
    Returns:
        List of routes for each batch
    """
    # Get parameters from args if provided, otherwise use defaults
    if args is not None:
        spoilage_penalty = args.spoilage_penalty if hasattr(args, 'spoilage_penalty') else 2.0
        pickup_bonus = args.pickup_bonus if hasattr(args, 'pickup_bonus') else 1.0
        dist_penalty = args.dist_penalty if hasattr(args, 'dist_penalty') else 1.0
    else:
        spoilage_penalty = 2.0
        pickup_bonus = 1.0
        dist_penalty = 1.0
    
    with Pool() as p:
        with tqdm(desc="Calling ORTools", total=data.batch_size) as pbar:
            results = [
                p.apply_async(
                    _solve_cp,
                    (nodes, data.veh_count, data.veh_capa, data.veh_speed, 
                     spoilage_penalty, pickup_bonus, dist_penalty),
                    callback=lambda _: pbar.update()
                ) for nodes in data.nodes_gen()
            ]
            routes = [res.get() for res in results]
            
            # Print first solution for debugging
            if routes and routes[0]:
                print("\nExample solution:")
                print_solution(routes[0], next(data.nodes_gen()), data.veh_speed)
                
    return routes