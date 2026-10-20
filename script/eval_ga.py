#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Evaluation: Genetic Algorithm vs OR-Tools
"""

from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ga_solve, ort_solve
from marpdan.utils import parse_args

import torch
import os
import time

# COST PARAMETERS 
COST_PARAMS = {
    'spoilage_penalty': 1,           # SPOILAGE_PENALTY
    'additional_late_penalty': 1,    # ADDITIONAL_LATE_PENALTY
    'unserved_penalty': 1,           # UNSERVED_PENALTY
    'idle_penalty_coef': 10,        # IDLE_PENALTY_COEF
    'dist_penalty_coef': 0.05,       # DIST_PENALTY_COEF
}


def compute_cost_like_environment(nodes, routes, veh_speed, veh_count, params, verbose=False):
    """
   
    Args:
        nodes: [num_nodes, 4] tensor (x, y, demand, spoilage_time)
        routes: list of vehicle routes [[route_v0], [route_v1], ...]
        veh_speed: vehicle speed
        veh_count: total number of vehicles
        params: dict with penalty coefficients
        verbose: print detailed breakdown
    
    Returns:
        total_cost: scalar cost value 
    """
    depot_pos = nodes[0, :2]
    total_cost = 0.0
    
    # Track which vehicles are used
    used_vehicles = set()
    served_nodes = set()
    
    # Track late deliveries at depot
    late_at_depot_nodes = set()
    
    
    step_distance_cost = 0.0
    step_spoilage_cost = 0.0
    
    for veh_idx, route in enumerate(routes):
        if not route:
            continue
        
        used_vehicles.add(veh_idx)
        current_pos = depot_pos.clone()
        current_time = 0.0
        
        pickup_info = []  # Store (node_idx, spoilage_time)
        
        for node_idx in route:
            served_nodes.add(node_idx)
            
            # Calculate travel to this node
            node_pos = nodes[node_idx, :2]
            dist = torch.norm(node_pos - current_pos).item()
            travel_time = dist / veh_speed
            arrival_time = current_time + travel_time
            
            # Distance penalty (applied during step)
            dist_cost = params['dist_penalty_coef'] * dist
            step_distance_cost += dist_cost
            total_cost += dist_cost
            
            # Pickup lateness check
            spoilage_time = nodes[node_idx, 3].item()
            direct_dist_to_depot = torch.norm(depot_pos - node_pos).item()
            direct_time_to_depot = direct_dist_to_depot / veh_speed
            latest_pickup_time = spoilage_time - direct_time_to_depot
            
            # Spoilage penalty if late pickup
            is_late_pickup = arrival_time > latest_pickup_time
            if is_late_pickup:
                spoil_cost = params['spoilage_penalty']
                step_spoilage_cost += spoil_cost
                total_cost += spoil_cost
            
            # Store for depot delivery check
            pickup_info.append((node_idx, spoilage_time))
            
            # Update position and time
            current_pos = node_pos
            current_time = arrival_time
        
        # Return to depot
        final_dist_to_depot = torch.norm(depot_pos - current_pos).item()
        depot_arrival_time = current_time + final_dist_to_depot / veh_speed
        
        # Check which nodes are delivered late to depot
        for node_idx, spoilage_time in pickup_info:
            if depot_arrival_time > spoilage_time:
                late_at_depot_nodes.add(node_idx)
    
    # 1. Late delivery penalty
    late_delivery_cost = params['additional_late_penalty'] * len(late_at_depot_nodes)
    total_cost += late_delivery_cost
    
    # 2. Unserved penalty
    all_nodes = set(range(1, len(nodes)))
    
    # Calculate infeasible nodes (same as environment)
    depot_pos_expand = nodes[0:1, :2]
    node_pos = nodes[:, :2]
    to_node_dist = torch.norm(node_pos - depot_pos_expand, dim=-1)
    round_trip_time = 2 * (to_node_dist / veh_speed)
    infeasible_mask = round_trip_time > nodes[:, 3]
    infeasible_nodes = set(torch.nonzero(infeasible_mask).flatten().tolist())
    infeasible_nodes.discard(0)
    
    unserved = (all_nodes - served_nodes) - infeasible_nodes
    unserved_cost = params['unserved_penalty'] * len(unserved)
    total_cost += unserved_cost
    
    # 3. Idle vehicles penalty
    idle_count = veh_count - len(used_vehicles)
    idle_cost = params['idle_penalty_coef'] * idle_count
    total_cost += idle_cost
    
    if verbose:
        print(f"\nCOST BREAKDOWN:")
        print(f"  Distance cost (steps):          {step_distance_cost:8.2f}")
        print(f"  Pickup lateness (steps):        {step_spoilage_cost:8.2f}")
        print(f"  Delivery lateness (episode):    {late_delivery_cost:8.2f}  ({len(late_at_depot_nodes)} nodes)")
        print(f"  Unserved patients (episode):    {unserved_cost:8.2f}  ({len(unserved)} nodes)")
        print(f"  Idle vehicles (episode):        {idle_cost:8.2f}  ({idle_count} vehicles)")
        print(f"  {'─'*40}")
        print(f"  TOTAL COST:                     {total_cost:8.2f}\n")
    
    return total_cost


def evaluate_routes_fast(routes, data, veh_count, params):
    """
     
    
    Args:
        routes: List of route sets (one per instance)
        data: PVRP dataset (normalized)
        veh_count: number of vehicles
        params: cost parameters dict
        
    Returns:
        torch.tensor of costs
    """
    costs = []
    
    print(f"  Evaluating {len(routes)} routes with fast cost calculation...")
    
    for b, route_set in enumerate(routes):
        if b % 100 == 0 and b > 0:
            print(f"    Instance {b}/{len(routes)}")
        
        nodes = data.nodes[b]
        
        verbose = (b < 2)  
        cost = compute_cost_like_environment(
            nodes,
            route_set,
            data.veh_speed,
            veh_count,
            params,
            verbose=verbose
        )
        
        costs.append(cost)
    
    print(f"  ✓ Evaluation complete")
    return torch.tensor(costs)


def get_cost_params_from_args(args):
   
    return {
        'spoilage_penalty': args.spoilage_penalty,
        'additional_late_penalty': args.additional_late_penalty,
        'unserved_penalty': args.unserved_penalty,
        'idle_penalty_coef': args.idle_penalty_coef,
        'dist_penalty_coef': args.dist_penalty_coef,
    }


def main(args):
    pb = args.problem_type
    n = args.customers_count
    m = args.vehicles_count

    out_dir = f"./results/{pb}_n{n}m{m}_baselines_{time.strftime('%y%m%d-%H%M')}/"
    os.makedirs(out_dir, exist_ok=True)
    data_path = f"./data/{pb}_n{n}m{m}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"

    print("=" * 80)
    print(f" FIXED Baseline Evaluation: {pb}_n{n}m{m}")
    print("=" * 80)
    print(f"Data: {data_path}")

    cost_params = get_cost_params_from_args(args)
    
    print(f"\n Cost Parameters (matching Environment):")
    for name, value in cost_params.items():
        print(f"   {name:25s} = {value}")

    # Load normalized data
    data = torch.load(data_path)
    print(f"\n Dataset: {data.nodes.shape[0]} instances")
    print(f"   Customers: {n}, Vehicles: {m}")

    # Create unnormalized dataset for solvers (they work in original scale)
    nodes_unnorm = data.nodes.clone()
    nodes_unnorm[:, :, :2] *= 100   # Unnormalize coordinates
    nodes_unnorm[:, :, 2] *= 200    # Unnormalize demand
    nodes_unnorm[:, :, 3] *= 480    # Unnormalize spoilage times

    unnormed_data = PVRP_Dataset(data.veh_count, data.veh_capa, data.veh_speed, nodes_unnorm)

    # GA parameters
    ga_population = getattr(args, 'ga_population', 100)
    ga_generations = getattr(args, 'ga_generations', 500)

    print(f"\n GA Parameters: population={ga_population}, generations={ga_generations}")

    # Solve with Genetic Algorithm
    print("\n" + "=" * 80)
    print(" Running Genetic Algorithm...")
    print("=" * 80)

    ga_start = time.time()
    ga_routes = ga_solve(unnormed_data, population_size=ga_population, generations=ga_generations)
    ga_solve_time = time.time() - ga_start

    print(f"\n✓ GA solving complete in {ga_solve_time:.2f}s ({ga_solve_time/data.nodes.shape[0]:.3f}s per instance)")

    #  FIXED: Fast cost calculation
    print("\n Evaluating GA routes with fast cost function...")
    ga_eval_start = time.time()
    ga_costs = evaluate_routes_fast(ga_routes, data, m, cost_params)
    ga_eval_time = time.time() - ga_eval_start
    ga_time = ga_solve_time + ga_eval_time

    print(f"\n GA Results:")
    print(f"  Solve time:  {ga_solve_time:>8.2f}s ({ga_solve_time/data.nodes.shape[0]:.3f}s per instance)")
    print(f"  Eval time:   {ga_eval_time:>8.2f}s ({ga_eval_time/data.nodes.shape[0]:.3f}s per instance)")
    print(f"  Total time:  {ga_time:>8.2f}s")
    print(f"  Mean cost:   {ga_costs.mean():>8.4f} ± {ga_costs.std():.4f}")
    print(f"  Range:       [{ga_costs.min():.4f}, {ga_costs.max():.4f}]")

    torch.save(ga_costs, out_dir + "ga.pyth")
    torch.save({"costs": ga_costs, "routes": ga_routes, 
                "solve_time": ga_solve_time, "eval_time": ga_eval_time, 
                "total_time": ga_time}, out_dir + "ga_full.pyth")

    # Solve with OR-Tools
    
    print("\n" + "=" * 80)
    print(" Running OR-Tools...")
    print("=" * 80)

    ort_start = time.time()
    ort_routes = ort_solve(unnormed_data)
    ort_solve_time = time.time() - ort_start

    print(f"\n✓ OR-Tools solving complete in {ort_solve_time:.2f}s ({ort_solve_time/data.nodes.shape[0]:.3f}s per instance)")

    # FIXED: Fast cost calculation
    print("\n Evaluating OR-Tools routes with fast cost function...")
    ort_eval_start = time.time()
    ort_costs = evaluate_routes_fast(ort_routes, data, m, cost_params)
    ort_eval_time = time.time() - ort_eval_start
    ort_time = ort_solve_time + ort_eval_time

    print(f"\n OR-Tools Results:")
    print(f"  Solve time:  {ort_solve_time:>8.2f}s ({ort_solve_time/data.nodes.shape[0]:.3f}s per instance)")
    print(f"  Eval time:   {ort_eval_time:>8.2f}s ({ort_eval_time/data.nodes.shape[0]:.3f}s per instance)")
    print(f"  Total time:  {ort_time:>8.2f}s")
    print(f"  Mean cost:   {ort_costs.mean():>8.4f} ± {ort_costs.std():.4f}")
    print(f"  Range:       [{ort_costs.min():.4f}, {ort_costs.max():.4f}]")

    torch.save(ort_costs, out_dir + "ort.pyth")
    torch.save({"costs": ort_costs, "routes": ort_routes,
                "solve_time": ort_solve_time, "eval_time": ort_eval_time,
                "total_time": ort_time}, out_dir + "ort_full.pyth")

    
    # Comparison Summary
    print("\n" + "=" * 80)
    print(" COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Method':<15} {'Mean Cost':>12} {'Std':>10} {'Total Time':>12} {'Time/Inst':>12}")
    print("-" * 65)
    print(f"{'GA':<15} {ga_costs.mean():>12.4f} {ga_costs.std():>10.4f} "
          f"{ga_time:>12.2f}s {ga_time/data.nodes.shape[0]:>11.3f}s")
    print(f"{'OR-Tools':<15} {ort_costs.mean():>12.4f} {ort_costs.std():>10.4f} "
          f"{ort_time:>12.2f}s {ort_time/data.nodes.shape[0]:>11.3f}s")

    # Gap calculation
    ga_vs_ort = (ga_costs.mean() - ort_costs.mean()) / ort_costs.mean() * 100
    print(f"\n GA vs OR-Tools gap: {ga_vs_ort:+.2f}%")

    # Per-instance comparison
    ga_wins = (ga_costs < ort_costs).sum().item()
    ort_wins = (ort_costs < ga_costs).sum().item()
    ties = (ga_costs == ort_costs).sum().item()
    
    print(f"\n Per-instance wins:")
    print(f"  GA wins:       {ga_wins}/{data.nodes.shape[0]} ({100*ga_wins/data.nodes.shape[0]:.1f}%)")
    print(f"  OR-Tools wins: {ort_wins}/{data.nodes.shape[0]} ({100*ort_wins/data.nodes.shape[0]:.1f}%)")
    print(f"  Ties:          {ties}/{data.nodes.shape[0]}")

    # Instance-by-instance gaps
    gaps = (ga_costs / ort_costs - 1) * 100
    print(f"\n Gap distribution:")
    print(f"  Mean: {gaps.mean():.2f}%")
    print(f"  Std:  {gaps.std():.2f}%")
    print(f"  Range: [{gaps.min():.2f}%, {gaps.max():.2f}%]")
    
    better_count = (gaps < -5).sum().item()
    similar_count = ((gaps >= -5) & (gaps <= 5)).sum().item()
    worse_count = (gaps > 5).sum().item()
    
    print(f"\n  GA much better (>5%): {better_count} ({100*better_count/len(gaps):.1f}%)")
    print(f"  Similar (±5%):        {similar_count} ({100*similar_count/len(gaps):.1f}%)")
    print(f"  GA much worse (>5%):  {worse_count} ({100*worse_count/len(gaps):.1f}%)")

    print(f"\n Results saved to: {out_dir}")
    
    print("\n" + "=" * 80)
    print(" USAGE NOTES")
    print("=" * 80)
    print("To compare with learned model:")
    print(f"  ga_costs = torch.load('{out_dir}ga.pyth')")
    print(f"  ort_costs = torch.load('{out_dir}ort.pyth')")
    print(f"  learned_greedy = torch.load('<results_dir>/mardan_greedy.pyth')")
    print(f"  learned_sample = torch.load('<results_dir>/mardan_sample100.pyth')")
    print("\n   CRITICAL: All costs computed using SAME cost function")
    print("     This ensures fair comparison across all methods!")
    print("\n  ⚡ PERFORMANCE: Fast cost calculation (~10s vs ~13min for 800 instances)")
    print("     Using direct computation instead of step-by-step simulation")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)