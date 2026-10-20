#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive PVRP Evaluation: All Methods Comparison
Compares: OR-Tools, Genetic Algorithm, RL Greedy, RL Sampling
Reports: Costs, Late pickups, Late deliveries, Unserved patients, Timing
"""

from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve, ga_solve
from marpdan.utils import *   # parse_args, load_old_weights, ...
from marpdan.dep import tqdm

import torch
import time
import os
from torch.utils.data import DataLoader
from itertools import repeat

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROLLOUTS = 100
BATCH_SIZE = 32

# ═══════════════════════════════════════════════════════════════════════════
# COST PARAMETERS - MUST MATCH YOUR Environment PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
COST_PARAMS = {
    'spoilage_penalty': 1,           # SPOILAGE_PENALTY
    'additional_late_penalty': 1,    # ADDITIONAL_LATE_PENALTY
    'unserved_penalty': 2,           # UNSERVED_PENALTY
    'idle_penalty_coef': .5,         # IDLE_PENALTY_COEF
    'dist_penalty_coef': 0.05,       # DIST_PENALTY_COEF
}

# Collected wall-clock timings, filled in during the run
TIMINGS = {}


def _make_env(batch_data):
    """Build a PVRP_Environment with the shared cost parameters."""
    return PVRP_Environment(
        batch_data,
        spoilage_penalty=COST_PARAMS['spoilage_penalty'],
        unserved_penalty=COST_PARAMS['unserved_penalty'],
        additional_late_penalty=COST_PARAMS['additional_late_penalty'],
        dist_penalty_coef=COST_PARAMS['dist_penalty_coef'],
        idle_penalty_coef=COST_PARAMS['idle_penalty_coef'],
    )


# ───────────────────────────────────────────────────────────────────────────
# Local copy of helper used by eval_apriori_routes (fix NameError)
# ───────────────────────────────────────────────────────────────────────────
def _pad_with_zeros(src_it):
    """Yield items from src iterable, then infinite zeros."""
    yield from src_it
    yield from repeat(0)


def eval_apriori_routes(dyna, routes, rollout_count=1):
    """
    Evaluate fixed (a-priori) routes by stepping Environment until done.
    Returns mean cost per instance.

    Route lists shorter than veh_count are padded with empty routes, so that
    a solver which leaves a vehicle idle does not cause an index error.
    """
    mean_cost = dyna.nodes.new_zeros(dyna.minibatch_size)
    veh_count = dyna.veh_count

    for _ in range(rollout_count):
        dyna.reset()
        routes_it = []
        for inst_routes in routes:
            iters = [_pad_with_zeros(route) for route in inst_routes]
            while len(iters) < veh_count:
                iters.append(_pad_with_zeros([]))
            routes_it.append(iters)

        rewards = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor(
                [[next(routes_it[n][i.item()])]
                 for n, i in enumerate(dyna.cur_veh_idx)],
                dtype=torch.int64
            )
            rewards.append(dyna.step(cust_idx))
        mean_cost += -torch.stack(rewards).sum(dim=0).squeeze(-1)

    return mean_cost / rollout_count


# ───────────────────────────────────────────────────────────────────────────
# Reporting (late/unserved) via route simulation (for stats only)
# ───────────────────────────────────────────────────────────────────────────
def compute_late_sets_from_routes(nodes, routes, veh_speed):
    """
    Route-simulation late statistics (for reporting only).
    - late_pickup: arrival at node > (spoilage - direct_time_to_depot)
    - late_depot: depot arrival after route end > spoilage
    - unserved: feasible nodes that never got served
    """
    late_pickup = set()
    late_depot = set()

    depot_pos = nodes[0, :2]
    all_nodes = set(range(1, len(nodes)))
    served_nodes = set()

    for route in routes:
        if not route:
            continue

        current_pos = depot_pos.clone()
        current_time = 0.0
        pickup_info = []

        for node_idx in route:
            served_nodes.add(node_idx)

            node_pos = nodes[node_idx, :2]
            dist = torch.norm(node_pos - current_pos).item()
            travel_time = dist / veh_speed
            arrival_time = current_time + travel_time

            spoilage_time = nodes[node_idx, 3].item()
            direct_dist_to_depot = torch.norm(depot_pos - node_pos).item()
            direct_time_to_depot = direct_dist_to_depot / veh_speed
            latest_pickup_time = spoilage_time - direct_time_to_depot

            if arrival_time > latest_pickup_time:
                late_pickup.add(node_idx)

            pickup_info.append((node_idx, spoilage_time))
            current_pos = node_pos
            current_time = arrival_time

        final_dist_to_depot = torch.norm(depot_pos - current_pos).item()
        depot_arrival_time = current_time + final_dist_to_depot / veh_speed

        for node_idx, spoilage_time in pickup_info:
            if depot_arrival_time > spoilage_time:
                late_depot.add(node_idx)

    # infeasible (never serviceable)
    depot_pos_expand = nodes[0:1, :2]
    node_pos = nodes[:, :2]
    to_node_dist = torch.norm(node_pos - depot_pos_expand, dim=-1)
    round_trip_time = 2 * (to_node_dist / veh_speed)
    infeasible_mask = round_trip_time > nodes[:, 3]
    infeasible_nodes = set(torch.nonzero(infeasible_mask).flatten().tolist())
    infeasible_nodes.discard(0)

    unserved = (all_nodes - served_nodes) - infeasible_nodes

    return late_pickup, late_depot, unserved


def extract_learned_routes_from_actions(actions, batch_idx, veh_count):
    """
    Extract per-instance routes from learner actions.
    Empty routes are kept so the returned list always has veh_count entries.
    """
    instance_routes = [[] for _ in range(veh_count)]
    for action in actions:
        veh_idx, node_idx = action
        v = veh_idx[batch_idx].item()
        n = node_idx[batch_idx].item()
        if n > 0:
            instance_routes[v].append(n)
    return instance_routes


# ───────────────────────────────────────────────────────────────────────────
# Baseline cost via stepping environment
# ───────────────────────────────────────────────────────────────────────────
def baseline_costs_via_env(data, all_routes, batch_indices, device=dev):
    """Compute costs for OR-Tools/GA by stepping PVRP_Environment."""
    batch_nodes = data.nodes[batch_indices].to(device)
    batch_mask = data.cust_mask[batch_indices].to(device) if data.cust_mask is not None else None

    batch_data = PVRP_Dataset(
        nodes=batch_nodes,
        cust_mask=batch_mask,
        veh_count=data.veh_count,
        veh_capa=data.veh_capa,
        veh_speed=data.veh_speed
    )

    env = _make_env(batch_data)
    batch_routes = [all_routes[i] for i in batch_indices.tolist()]

    return eval_apriori_routes(env, batch_routes, rollout_count=1).detach().cpu()


def evaluate_baseline_with_dataloader(data, routes, method_name):
    """
    Baseline evaluation for OR-Tools / GA:
    - Stats (late/unserved): route simulation (reporting)
    - Costs: environment stepping (matches env.step)
    """
    stats = {'late_pickup': 0, 'late_delivery': 0, 'unserved': 0, 'costs': []}

    print("\n" + "=" * 80)
    print(f"EVALUATING {method_name.upper()} SOLUTIONS...")
    print("=" * 80)

    idx_loader = DataLoader(range(data.nodes.size(0)), batch_size=BATCH_SIZE, shuffle=False)

    for batch_indices in tqdm(idx_loader, desc=method_name):
        batch_costs = baseline_costs_via_env(data, routes, batch_indices, device=dev)

        for local_idx, global_idx in enumerate(batch_indices.tolist()):
            nodes = data.nodes[global_idx]
            route_set = routes[global_idx]

            late_p, late_d, unserved = compute_late_sets_from_routes(nodes, route_set, data.veh_speed)

            stats['late_pickup'] += len(late_p)
            stats['late_delivery'] += len(late_d - late_p)  # reporting only
            stats['unserved'] += len(unserved)
            stats['costs'].append(batch_costs[local_idx].item())

    return stats


# ───────────────────────────────────────────────────────────────────────────
# RL evaluation
# ───────────────────────────────────────────────────────────────────────────
def evaluate_greedy_with_dataloader(data, learner):
    """RL Greedy evaluation (costs from env reward via learner)."""
    stats = {'late_pickup': 0, 'late_delivery': 0, 'unserved': 0, 'costs': []}

    print("\n" + "=" * 80)
    print("EVALUATING RL GREEDY...")
    print("=" * 80)

    learner.greedy = True
    n_inst = data.nodes.size(0)
    idx_loader = DataLoader(range(n_inst), batch_size=BATCH_SIZE, shuffle=False)

    t0 = time.time()
    for batch_indices in tqdm(idx_loader, desc="Greedy"):
        batch_nodes = data.nodes[batch_indices].to(dev)
        batch_mask = data.cust_mask[batch_indices].to(dev) if data.cust_mask is not None else None

        batch_data = PVRP_Dataset(
            nodes=batch_nodes,
            cust_mask=batch_mask,
            veh_count=data.veh_count,
            veh_capa=data.veh_capa,
            veh_speed=data.veh_speed
        )

        env = _make_env(batch_data)

        with torch.no_grad():
            actions, _, rewards = learner(env)

        costs_batch = torch.stack(rewards).sum(0).mul(-1).squeeze(1).detach().cpu()

        for local_idx, global_idx in enumerate(batch_indices.tolist()):
            nodes_cpu = data.nodes[global_idx]
            routes = extract_learned_routes_from_actions(actions, local_idx, data.veh_count)

            late_p, late_d, unserved = compute_late_sets_from_routes(nodes_cpu, routes, data.veh_speed)

            stats['late_pickup'] += len(late_p)
            stats['late_delivery'] += len(late_d - late_p)
            stats['unserved'] += len(unserved)
            stats['costs'].append(costs_batch[local_idx].item())

    elapsed = time.time() - t0
    TIMINGS['greedy'] = (elapsed, 1000.0 * elapsed / n_inst)
    print(f"   Greedy total: {elapsed:.1f}s  ({1000.0*elapsed/n_inst:.2f} ms per instance)")

    return stats


def evaluate_sampling_single_instance(data, learner, instance_idx):
    """
    Best-of-N sampling for one instance.

    The instance is replicated ROLLOUTS times along the batch dimension so
    that all candidate trajectories are drawn in a single forward pass,
    instead of ROLLOUTS separate calls with batch size 1.
    """
    nodes = data.nodes[instance_idx]

    rep_nodes = nodes.unsqueeze(0).expand(ROLLOUTS, -1, -1).contiguous().to(dev)
    if data.cust_mask is not None:
        rep_mask = data.cust_mask[instance_idx].unsqueeze(0) \
                       .expand(ROLLOUTS, -1).contiguous().to(dev)
    else:
        rep_mask = None

    rep_data = PVRP_Dataset(
        nodes=rep_nodes,
        cust_mask=rep_mask,
        veh_count=data.veh_count,
        veh_capa=data.veh_capa,
        veh_speed=data.veh_speed
    )

    env = _make_env(rep_data)

    learner.greedy = False
    with torch.no_grad():
        actions, _, rewards = learner(env)

    costs = torch.stack(rewards).sum(0).mul(-1).squeeze(-1).detach().cpu()
    best_idx = int(costs.argmin().item())
    best_cost = float(costs[best_idx].item())
    best_route = extract_learned_routes_from_actions(actions, best_idx, data.veh_count)

    return best_route, best_cost


def evaluate_sampling_with_dataloader(data, learner):
    """RL Sampling evaluation over all instances."""
    stats = {'late_pickup': 0, 'late_delivery': 0, 'unserved': 0, 'costs': []}

    print("\n" + "=" * 80)
    print("EVALUATING RL SAMPLING...")
    print("=" * 80)

    n_inst = data.nodes.size(0)

    t0 = time.time()
    for global_idx in tqdm(range(n_inst), desc="Sampling"):
        best_route, best_cost = evaluate_sampling_single_instance(data, learner, global_idx)

        nodes = data.nodes[global_idx]
        late_p, late_d, unserved = compute_late_sets_from_routes(nodes, best_route, data.veh_speed)

        stats['late_pickup'] += len(late_p)
        stats['late_delivery'] += len(late_d - late_p)
        stats['unserved'] += len(unserved)
        stats['costs'].append(best_cost)

    elapsed = time.time() - t0
    TIMINGS['sampling'] = (elapsed, 1000.0 * elapsed / n_inst)
    print(f"   Sampling total: {elapsed:.1f}s  ({1000.0*elapsed/n_inst:.1f} ms per instance)")

    return stats


# ───────────────────────────────────────────────────────────────────────────
# Reporting
# ───────────────────────────────────────────────────────────────────────────
def print_comprehensive_comparison_report(results, total_patients, num_instances):
    """Comprehensive comparison report."""
    print("\n" + "=" * 100)
    print(" COMPREHENSIVE EVALUATION - ALL METHODS COMPARISON ".center(100, "="))
    print("=" * 100)

    print(f"\nTotal Test Instances: {num_instances}")
    print(f"Total Patients: {total_patients}")
    print(f"Patients per Instance: {total_patients // num_instances}")

    print("\nCost Parameters (matching Environment):")
    for key, value in COST_PARAMS.items():
        print(f"  {key:25s} = {value}")

    all_methods = ['ortools', 'ga', 'greedy', 'sampling']
    available_methods = [m for m in all_methods if m in results]

    name_of = {'ortools': 'OR-Tools', 'ga': 'Genetic Algorithm',
               'greedy': 'RL Greedy', 'sampling': 'RL Sampling'}

    for method in available_methods:
        name = name_of[method]
        d = results[method]

        avg_cost = sum(d['costs']) / len(d['costs'])
        std_cost = torch.tensor(d['costs']).std().item()

        print(f"\n{'─'*100}")
        print(f" {name} ".center(100, "─"))
        print(f"{'─'*100}")
        print(f"  Average Cost:        {avg_cost:7.2f} ± {std_cost:5.2f}")
        print(f"  Late Pickups:        {d['late_pickup']:5d}  ({100*d['late_pickup']/total_patients:6.2f}%)")
        print(f"  Late Deliveries:     {d['late_delivery']:5d}  ({100*d['late_delivery']/total_patients:6.2f}%)")
        print(f"  Unserved Patients:   {d['unserved']:5d}  ({100*d['unserved']/total_patients:6.2f}%)")
        print(f"  Total Violations:    {d['late_pickup'] + d['late_delivery'] + d['unserved']:5d}")

    print("\n" + "=" * 100)
    print(" SUMMARY TABLE: ALL METHODS ".center(100, "="))
    print("=" * 100)

    print(f"\n{'Method':<20} {'Avg Cost':>12} {'Std':>10} {'Late P':>8} {'Late D':>8} {'Unserved':>10} {'Total Viol':>12}")
    print("─" * 100)

    for method in available_methods:
        d = results[method]
        avg_cost = sum(d['costs']) / len(d['costs'])
        std_cost = torch.tensor(d['costs']).std().item()
        total_viol = d['late_pickup'] + d['late_delivery'] + d['unserved']

        print(f"{name_of[method]:<20} {avg_cost:>12.4f} {std_cost:>10.4f} "
              f"{d['late_pickup']:>8d} {d['late_delivery']:>8d} {d['unserved']:>10d} {total_viol:>12d}")

    # ── Timing summary ──
    print("\n" + "=" * 100)
    print(" TIMING SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"\n{'Method':<20} {'Total (s)':>12} {'Per instance (ms)':>20} {'Execution':>18}")
    print("─" * 100)

    exec_mode = {
        'ortools':  'parallel (Pool)',
        'ga':       'parallel (Pool)',
        'greedy':   'sequential',
        'sampling': 'sequential',
    }
    for key in ['ortools', 'ga', 'greedy', 'sampling']:
        if key in TIMINGS:
            total_s, per_ms = TIMINGS[key]
            print(f"{name_of[key]:<20} {total_s:>12.1f} {per_ms:>20.2f} {exec_mode[key]:>18}")

    print("\nNote: OR-Tools and GA are solved across all CPU cores via multiprocessing.Pool,")
    print("      while RL inference runs in a single process. Wall-clock times are therefore")
    print("      not directly comparable without accounting for this difference.")

    print("\n" + "=" * 100 + "\n")


def main(args):
    pb = args.problem_type
    n = args.customers_count
    m = args.vehicles_count
    epoch = args.epoch_count
    date = "260901-2039"

    out_dir = f"./results/{pb}_n{n}m{m}_comprehensive_{time.strftime('%y%m%d-%H%M')}/"
    os.makedirs(out_dir, exist_ok=True)

    data_path = f"./data/{pb}_n{n}m{m}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"
    model_path = f"./output/PVRPn{n}m{m}_{date}/chkpt_ep{epoch}.pyth"

    print(f"\n{'='*100}")
    print(f" COMPREHENSIVE PVRP EVALUATION: {pb}_n{n}m{m} ".center(100))
    print(f"{'='*100}\n")

    print(f"Loading data from: {data_path}")
    print(f"Loading model from: {model_path}\n")

    try:
        data = torch.load(data_path)

        num_instances = data.nodes.size(0)
        num_customers = data.nodes.size(1) - 1
        total_patients = num_instances * num_customers

        print(f"num_instances = {num_instances}")
        print(f"dataloader batches = {(num_instances + BATCH_SIZE - 1)//BATCH_SIZE}")

        print("Dataset Info:")
        print(f"  Instances: {num_instances}")
        print(f"  Customers per instance: {num_customers}")
        print(f"  Vehicles: {data.veh_count}")
        print(f"  Total patients: {total_patients}\n")

        # ───────────────────────────────────────────────────────────────
        # Baseline solving on unnormalized copy
        # ───────────────────────────────────────────────────────────────
        print("=" * 80)
        print(" GENERATING BASELINE SOLUTIONS...")
        print("=" * 80)

        nodes_unnorm = data.nodes.clone()
        nodes_unnorm[:, :, :2] *= 100
        nodes_unnorm[:, :, 3] *= 480

        unnormed_data = PVRP_Dataset(data.veh_count, data.veh_capa, data.veh_speed, nodes_unnorm)

        print("\n1. Solving with OR-Tools...")
        ort_start = time.time()
        ref_routes = ort_solve(unnormed_data)
        ort_time = time.time() - ort_start
        TIMINGS['ortools'] = (ort_time, 1000.0 * ort_time / num_instances)
        print(f"   OR-Tools complete in {ort_time:.2f}s ({ort_time/num_instances:.3f}s per instance)")

        print("\n2. Solving with Genetic Algorithm...")
        ga_population = getattr(args, 'ga_population', 100)
        ga_generations = getattr(args, 'ga_generations', 500)
        print(f"   Parameters: population={ga_population}, generations={ga_generations}")

        ga_start = time.time()
        ga_routes = ga_solve(unnormed_data, population_size=ga_population, generations=ga_generations)
        ga_time = time.time() - ga_start
        TIMINGS['ga'] = (ga_time, 1000.0 * ga_time / num_instances)
        print(f"   GA complete in {ga_time:.2f}s ({ga_time/num_instances:.3f}s per instance)")

        # normalize after baseline solving
        data.normalize()

        # Load RL model
        print("\n3. Loading RL model...")
        learner = AttentionLearner(
            cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
            veh_state_size=PVRP_Environment.VEH_STATE_SIZE
        )
        chkpt = torch.load(model_path, map_location="cpu")
        load_old_weights(learner, chkpt["model"])
        learner.to(dev)
        learner.eval()
        print("   RL model loaded\n")

        # Evaluate
        results = {}
        results['ortools'] = evaluate_baseline_with_dataloader(data, ref_routes, "OR-Tools")
        results['ga'] = evaluate_baseline_with_dataloader(data, ga_routes, "GA")
        results['greedy'] = evaluate_greedy_with_dataloader(data, learner)
        results['sampling'] = evaluate_sampling_with_dataloader(data, learner)

        print_comprehensive_comparison_report(results, total_patients, num_instances)

        save_path = out_dir + "comprehensive_results.pyth"
        torch.save({'results': results, 'timings': TIMINGS}, save_path)
        print(f"Results saved to: {save_path}\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main(parse_args())