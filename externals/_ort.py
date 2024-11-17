from marpdan.dep import ORTOOLS_ENABLED, pywrapcp, routing_enums_pb2
from marpdan.dep import tqdm

from multiprocessing import Pool

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total load of all routes: {total_load}")


def _solve_cp(nodes, veh_count, veh_capa, veh_speed, late_cost):

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(nodes.size(0), veh_count, 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return nodes[from_node, :2].sub(nodes[to_node, :2]).pow(2).sum().pow(0.5)
   

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return nodes[from_node,2]


    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [veh_capa for _ in range(veh_count)],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    '''search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)
    '''
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    routes = []
    for i in range(veh_count):
        route = []
        idx = routing.Start(i)
        while not routing.IsEnd(idx):
            idx = solution.Value(routing.NextVar(idx))
            route.append( manager.IndexToNode(idx) )
        routes.append(route)

    return routes
'''    
def ort_solve(data, late_cost=1):
    routes = []
    with tqdm(desc="Calling ORTools", total=data.batch_size) as pbar:
        for nodes in data.nodes_gen():
            route_temp = _solve_cp(nodes, data.veh_count, data.veh_capa, data.veh_speed, late_cost)
            routes.append(route_temp)
            pbar.update()
    return routes'''
 
def ort_solve(data, late_cost = 1):
    with Pool() as p:
        with tqdm(desc = "Calling ORTools", total = data.batch_size) as pbar:
            results = [
                p.apply_async(_solve_cp, (
                    nodes, 
                    data.veh_count, 
                    data.veh_capa, 
                    data.veh_speed, 
                    late_cost
                    ),
                callback = lambda _:pbar.update()
                ) for nodes in data.nodes_gen()]
            routes = [res.get() for res in results]
    return routes
