#ort.py 8 october
from marpdan.dep import ORTOOLS_ENABLED, pywrapcp, routing_enums_pb2
from marpdan.dep import tqdm
from multiprocessing import Pool
from marpdan.utils import parse_args

def _solve_cp(nodes, veh_count, veh_capa, veh_speed, spoilage_penalty):
    """Solve PVRP instance using OR-Tools with correct penalty coefficients
    
    Objective weights (matching the mathematical model):
    - Distance: 0.05
    - Pickup lateness: 1.0
    - Delivery lateness: 1.0  
    - Idle vehicles: 10.0
    """
    
    # ضرایب جریمه از مدل ریاضی
    ALPHA_DISTANCE = 0.05      # α₂: جریمه مسافت
    ALPHA_PICKUP_LATE = 1.0    # α₁: جریمه دیرکرد pickup
    ALPHA_DELIVERY_LATE = 1.0  # α₃: جریمه دیرکرد delivery
    ALPHA_IDLE = 10.0          # α₅: جریمه ماشین بیکار
    
    # Scale factor برای تبدیل penalties به واحد distance
    # چون OR-Tools همه چیز را به صورت cost می‌بیند
    DISTANCE_SCALE = int(1.0 / ALPHA_DISTANCE)  # = 20
    PICKUP_PENALTY_SCALED = int(ALPHA_PICKUP_LATE * DISTANCE_SCALE)  # = 20
    DELIVERY_PENALTY_SCALED = int(ALPHA_DELIVERY_LATE * DISTANCE_SCALE)  # = 20
    IDLE_PENALTY_SCALED = int(ALPHA_IDLE * DISTANCE_SCALE)  # = 200
    
    manager = pywrapcp.RoutingIndexManager(nodes.size(0), veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # محاسبه زمان برگشت به دیپو برای هر گره (پیش از callbacks)
    depot_pos = nodes[0, :2]
    depot_return_times = []
    for i in range(nodes.size(0)):
        node_pos = nodes[i, :2]
        dist_to_depot = (node_pos - depot_pos).pow(2).sum().pow(0.5)
        return_time = int(dist_to_depot / veh_speed)
        depot_return_times.append(return_time)

    # Distance callback - بدون scaling چون بعداً در cost evaluator scale می‌کنیم
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(nodes[from_node, :2].sub(nodes[to_node, :2]).pow(2).sum().pow(0.5))

    dist_callback_idx = routing.RegisterTransitCallback(distance_callback)
    
    # ★ Arc Cost با ضریب 0.05 برای distance
    def arc_cost_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        base_distance = distance_callback(from_index, to_index)
        
        # ★ Cost = 0.05 × distance (در واحد integer)
        # برای حفظ دقت، distance را در 100 ضرب می‌کنیم و سپس در 5 ضرب می‌کنیم
        # یعنی: cost = distance × 5 / 100 = distance × 0.05
        distance_cost = base_distance  # بعداً با SetFixedCostOfAllVehicles handle می‌شود
        
        # اگر to_node یک patient است (نه depot)
        if to_node > 0:
            survival_time = int(nodes[to_node, 3].item())
            
            # ★ چک کردن اینکه آیا رفتن به این گره باعث مرگ حتمی می‌شود
            # حتی با رفتن مستقیم از depot
            min_time_to_save = depot_return_times[to_node] * 2
            
            if min_time_to_save >= survival_time:
                # این گره اصلاً نجات‌پذیر نیست - penalty خیلی بالا
                # باید آنقدر بالا باشد که هرگز انتخاب نشود
                return distance_cost + int(1e8)
        
        return distance_cost
    
    arc_cost_idx = routing.RegisterTransitCallback(arc_cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(arc_cost_idx)

    # Time callback
    def time_callback(from_index, to_index):
        return int(distance_callback(from_index, to_index) / veh_speed)

    time_callback_idx = routing.RegisterTransitCallback(time_callback)

    # Add Time dimension
    max_time = int(nodes[:, 3].max().item() * 2)  # افزایش برای اطمینان
    routing.AddDimension(
        time_callback_idx,
        0,  # no slack
        max_time,
        True,
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # Add capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return 1 if from_node != 0 else 0

    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx,
        0,
        [veh_capa] * veh_count,
        True,
        "Capacity"
    )

    # محاسبه زمان برگشت به دیپو برای هر گره
    depot_pos = nodes[0, :2]
    depot_return_times = []
    for i in range(nodes.size(0)):
        node_pos = nodes[i, :2]
        dist_to_depot = (node_pos - depot_pos).pow(2).sum().pow(0.5)
        return_time = int(dist_to_depot / veh_speed)
        depot_return_times.append(return_time)

    # تشخیص گره‌های infeasible و optional
    infeasible_nodes = []
    feasible_nodes = []
    
    for node in range(1, nodes.size(0)):
        index = manager.NodeToIndex(node)
        survival_time = int(nodes[node, 3].item())
        
        # زمان حداقل برای نجات: رفت + برگشت (رفتن مستقیم)
        min_time_to_save = depot_return_times[node] * 2
        
        # ★ بررسی infeasibility: آیا حتی با رفتن مستقیم هم قابل نجات نیست؟
        if min_time_to_save >= survival_time:
            # این گره infeasible است - اصلاً نمی‌توان آن را نجات داد
            infeasible_nodes.append(node)
            # با penalty خیلی خیلی بالا optional می‌کنیم تا هرگز انتخاب نشود
            routing.AddDisjunction([index], int(1e8))
            print(f" Node {node}: INFEASIBLE (min_time={min_time_to_save}, survival={survival_time})")
            continue
        
        # ★ محاسبه latest pickup time با margin امنیتی
        # latest_pickup = زمانی که باید pick شود تا survival نقض نشود
        # اما ما باید در نظر بگیریم که ماشین ممکن است به گره‌های دیگر هم برود
        # پس یک margin امنیتی اضافه می‌کنیم
        
        # محافظه‌کارانه: فرض کنیم ماشین حداکثر به veh_capa گره می‌رود
        # و میانگین زمان اضافی برای هر گره = depot_return_time / 2
        # این یک تخمین محافظه‌کارانه است
        avg_detour_per_node = depot_return_times[node] * 0.3  # 30% هر گره
        max_detour = avg_detour_per_node * min(veh_capa - 1, 3)  # حداکثر 3 گره دیگر
        
        latest_pickup = survival_time - depot_return_times[node] - max_detour
        
        if latest_pickup <= 0:
            # این گره هم infeasible است
            infeasible_nodes.append(node)
            # Penalty برای drop کردن = 10 (idle vehicle penalty)
            # چون drop کردن یک گره مثل idle ماندن یک ظرفیت است
            routing.AddDisjunction([index], IDLE_PENALTY_SCALED)
            print(f" Node {node}: INFEASIBLE (latest_pickup={latest_pickup:.1f}<=0, detour={max_detour:.1f})")
            continue
        
        feasible_nodes.append(node)
        
        # ★ تنظیم time window برای pickup - با محدودیت محافظه‌کارانه
        time_dimension.CumulVar(index).SetMax(int(latest_pickup))
        
        # ★ اضافه کردن soft constraint با penalty برای pickup lateness
        # Penalty = 1.0 (scaled = 20)
        if spoilage_penalty > 0:
            # Soft constraint: اجازه نقض با penalty
            time_dimension.SetCumulVarSoftUpperBound(
                index,
                int(latest_pickup),
                PICKUP_PENALTY_SCALED  # α₁ = 1.0 (scaled)
            )
            # گره را optional می‌کنیم
            # اگر drop شود = مثل یک idle vehicle
            routing.AddDisjunction([index], IDLE_PENALTY_SCALED)
        else:
            # Hard constraint: اصلاً نمی‌تواند نقض شود
            # گره را optional می‌کنیم با penalty خیلی بالا
            routing.AddDisjunction([index], int(1e8))

    # ★ اضافه کردن constraint برای delivery time
    # برای هر گره، باید تضمین کنیم که زمان برگشت به دیپو از survival تجاوز نکند
    # این کار را با یک dimension جدید انجام می‌دهیم که "وقت برگشت تا دیپو" را track می‌کند
    
    #  محدودیت OR-Tools: نمی‌توانیم مستقیماً delivery lateness (L_j^h) را penalize کنیم
    # چون OR-Tools فقط یک Time dimension دارد که arrival time را track می‌کند
    # برای delivery time باید arrival_time + return_time_to_depot را محاسبه کنیم
    # که این در حین optimization قابل محاسبه نیست
    
    # راه‌حل: ما با time window محافظه‌کارانه (با safety margin) تضمین می‌کنیم
    # که delivery lateness به حداقل برسد. سپس در post-processing آن را check می‌کنیم.
    
    # برای هر vehicle، maximum time را محدود می‌کنیم
    for vehicle_id in range(veh_count):
        end_index = routing.End(vehicle_id)
        time_dimension.CumulVar(end_index).SetMax(max_time)
        
        # ★ Penalty برای ماشین‌های بیکار (α₅ = 10)
        # این در objective function implicit است چون:
        # - اگر ماشین بیکار باشد، هیچ گره‌ای serve نمی‌کند
        # - گره‌های drop شده penalty دارند (IDLE_PENALTY_SCALED)
    
    # ★ نکته مهم: ضرایب alpha در مدل ریاضی:
    # α₁ = 1.0  : Pickup lateness (L_j^p) - handled by SetCumulVarSoftUpperBound
    # α₂ = 0.05 : Distance - handled by arc cost
    # α₃ = 1.0  : Delivery lateness (L_j^h) - partially handled by safety margin
    # α₄ = 1.0    : Unserved patients (U_j) - handled by AddDisjunction
    # α₅ = 10.0 : Idle vehicles (I_v) - handled by AddDisjunction penalty

    # Solver settings
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    )
    search_parameters.time_limit.FromSeconds(60)
    search_parameters.solution_limit = 100

    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        print("No solution found!")
        return []

    # Extract routes و بررسی صحت با اعمال post-processing
    routes = []
    total_violations = 0
    dropped_nodes = []
    
    for vehicle_id in range(veh_count):
        route = []
        index = routing.Start(vehicle_id)
        
        # Track cumulative time و survival check
        cumulative_time = 0
        violations_in_route = []
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            
            if node_idx != 0:
                # بررسی survival time به صورت دقیق
                arrival_time = solution.Value(time_dimension.CumulVar(index))
                survival_time = int(nodes[node_idx, 3].item())
                
                # ★ محاسبه دقیق delivery time
                # باید زمان برگشت از END of route را محاسبه کنیم
                # برای الان، تخمین می‌زنیم با return time مستقیم
                return_time = depot_return_times[node_idx]
                
                # Estimate: زمان delivery = زمان فعلی + زمان برگشت
                # (این تخمین محافظه‌کارانه است)
                estimated_delivery = arrival_time + return_time
                
                # اگر نقض survival رخ داد
                if estimated_delivery > survival_time:
                    violations_in_route.append({
                        'node': node_idx,
                        'arrival': arrival_time,
                        'delivery': estimated_delivery,
                        'survival': survival_time,
                        'violation': estimated_delivery - survival_time
                    })
                    total_violations += 1
                    
                    print(f" Vehicle {vehicle_id}, Node {node_idx}: SURVIVAL VIOLATION!")
                    print(f"   Arrival: {arrival_time}, Est.Delivery: {estimated_delivery}, "
                          f"Survival: {survival_time}, Violation: {estimated_delivery - survival_time}")
                
                route.append(node_idx)
                
            index = solution.Value(routing.NextVar(index))
        
        # ★ Post-processing: حذف گره‌هایی که نقض جدی دارند
        # if violations_in_route and spoilage_penalty == 0:
        #     # در حالت hard constraint، نباید violation داشته باشیم
        #     print(f" Route {vehicle_id} has {len(violations_in_route)} violations - این نباید اتفاق بیفتد!")
        
        if route:
            routes.append(route)
    
    # بررسی گره‌های drop شده
    served_nodes = set()
    for route in routes:
        served_nodes.update(route)
    
    all_nodes = set(range(1, nodes.size(0)))
    dropped_nodes = all_nodes - served_nodes
    
    # print(f"\n Solution Stats:")
    # print(f"   Total routes: {len(routes)}")
    # print(f"   Served nodes: {len(served_nodes)}/{len(all_nodes)}")
    # print(f"   Dropped nodes: {len(dropped_nodes)} - {sorted(dropped_nodes)}")
    # print(f"   Infeasible nodes: {len(infeasible_nodes)} - {sorted(infeasible_nodes)}")
    # print(f"   Survival violations: {total_violations}")
    
    return routes


def print_solution(routes, nodes, veh_speed):
    """Print readable solution with violation checks"""
    depot_pos = nodes[0, :2]
    total_distance = 0
    total_time = 0
    total_violations = 0
    
    for i, route in enumerate(routes):
        if not route:
            continue
            
        print(f"\n Route {i}:")
        time = 0
        distance = 0
        current_pos = depot_pos
        
        print(f"  Depot → ", end='')
        for node in route:
            next_pos = nodes[node, :2]
            dist = (next_pos - current_pos).pow(2).sum().pow(0.5)
            travel_time = dist / veh_speed
            
            distance += dist
            time += travel_time
            
            # بررسی survival
            survival = nodes[node, 3].item()
            return_dist = (depot_pos - next_pos).pow(2).sum().pow(0.5)
            return_time = return_dist / veh_speed
            delivery_time = time + return_time
            
            
            if delivery_time > survival:
                total_violations += 1
            
            current_pos = next_pos
        
        # Return to depot
        dist = (depot_pos - current_pos).pow(2).sum().pow(0.5)
        distance += dist
        time += dist / veh_speed
        print(f"Depot(t={time:.1f})")
        
        print(f"  Distance: {distance:.1f},  Time: {time:.1f}")
        
        total_distance += distance
        total_time += time
    
    print(f"\n Total Summary:")
    print(f"  Total distance: {total_distance:.1f}")
    print(f"  Total time: {total_time:.1f}")
    print(f"  Survival violations: {total_violations}")


def ort_solve(data):
    args = parse_args()
    spoilage_penalty = args.spoilage_penalty
    
    with Pool() as p:
        with tqdm(desc="Calling ORTools", total=data.batch_size) as pbar:
            results = [
                p.apply_async(
                    _solve_cp,
                    (nodes, data.veh_count, data.veh_capa, data.veh_speed, spoilage_penalty),
                    callback=lambda _: pbar.update()
                ) for nodes in data.nodes_gen()
            ]
            routes = [res.get() for res in results]
            
            # if routes and routes[0]:
            #     print("\n" + "="*60)
            #     print("Example Solution Analysis:")
            #     print("="*60)
            #     print_solution(routes[0], next(data.nodes_gen()), data.veh_speed)
                
    return routes