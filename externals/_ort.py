from marpdan.dep import ORTOOLS_ENABLED, pywrapcp, routing_enums_pb2
from marpdan.dep import tqdm
from marpdan.utils import eval_apriori_routes
import torch
from concurrent.futures import ProcessPoolExecutor, TimeoutError

def _solve_cp(nodes, veh_count, veh_capa, veh_speed, spoilage_penalty):
    """Solve PVRP instance with enhanced perishability constraints"""
    # Convert to numpy for OR-Tools compatibility
    nodes_np = nodes.cpu().numpy()
    manager = pywrapcp.RoutingIndexManager(nodes_np.shape[0], veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance matrix calculation
    def distance_matrix(from_idx, to_idx):
        return int(torch.norm(nodes[from_idx,:2] - nodes[to_idx,:2]).item() * 1000)

    dist_callback_idx = routing.RegisterTransitCallback(distance_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback_idx)

    # Time constraints with vehicle speed
    def time_callback(from_idx, to_idx):
        return int(distance_matrix(from_idx, to_idx) / veh_speed * 1000)

    time_callback_idx = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_idx,
        0,  # no slack
        int(1e7),  # large enough maximum time
        True, 
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Perishability constraints
    for node in range(1, nodes_np.shape[0]):
        idx = manager.NodeToIndex(node)
        deadline = int(nodes[node,3].item() * 1000)  # assume normalized time
        time_dim.CumulVar(idx).SetMax(deadline)
        time_dim.SetCumulVarSoftUpperBound(idx, deadline, int(spoilage_penalty * 1000))

    # Capacity constraints
    def demand_callback(from_idx):
        return 1 if manager.IndexToNode(from_idx) != 0 else 0

    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx,
        0,
        [veh_capa] * veh_count,
        True,
        "Capacity"
    )

    # Solver configuration
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = 30
    search_params.log_search = False

    try:
        solution = routing.SolveWithParameters(search_params)
        routes = []
        for vid in range(veh_count):
            route = []
            idx = routing.Start(vid)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0: route.append(node)
                idx = solution.Value(routing.NextVar(idx))
            if route: routes.append(route)
        return routes
    except Exception:
        return []

def ort_solve(data, spoilage_penalty, timeout=60):
    """Parallel OR-Tools solver with proper device handling"""
    if not ORTOOLS_ENABLED:
        raise RuntimeError("OR-Tools not available")
    
    # Ensure CPU computation for OR-Tools
    device = data.nodes.device
    cpu_data = data.nodes.cpu()
    
    with ProcessPoolExecutor() as executor:
        futures = []
        for batch_nodes in cpu_data:
            futures.append(executor.submit(
                _solve_cp,
                batch_nodes,
                data.veh_count,
                data.veh_capa,
                data.veh_speed,
                spoilage_penalty
            ))
        
        routes = []
        with tqdm(total=len(futures), desc="OR-Tools Solving") as pbar:
            for future in futures:
                try:
                    routes.append(future.result(timeout))
                except Exception:
                    routes.append([])
                pbar.update(1)
                
    return [r for r in routes if r]