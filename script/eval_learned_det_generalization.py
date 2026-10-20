#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Zero-Shot Generalization Evaluation
Evaluates a model trained on (n1, m1) on test instances of size (n2, m2).
Compares: OR-Tools, Genetic Algorithm, RL Greedy, RL Sampling.

Costs for every method are computed by stepping PVRP_Environment, so they
match the training objective exactly.
"""

from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve, ga_solve
from marpdan.utils import *
from marpdan.dep import tqdm

import torch
import time
import os
import glob
from torch.utils.data import DataLoader
from itertools import repeat

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROLLOUTS = 100
BATCH_SIZE = 32

COST_PARAMS = {
    'spoilage_penalty': 1,
    'additional_late_penalty': 1,
    'unserved_penalty': 2,
    'idle_penalty_coef': 0.5,
    'dist_penalty_coef': 0.05,
}

# Wall-clock timings collected during the run
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
        logger=None,
    )


# ───────────────────────────────────────────────────────────────────────────
# Model and checkpoint discovery
# ───────────────────────────────────────────────────────────────────────────
def find_model_directory_by_date(base_pattern, date=None):
    if date:
        matching = glob.glob(f"{base_pattern}{date}")
        if not matching:
            raise FileNotFoundError(f"No directory found matching: {base_pattern}{date}")
        return matching[0]

    matching = glob.glob(f"{base_pattern}*")
    if not matching:
        raise FileNotFoundError(f"No directories found matching: {base_pattern}*")
    matching.sort()
    print(f"Found latest directory: {matching[-1]}")
    return matching[-1]


def find_latest_checkpoint(output_dir, epoch=None):
    if epoch is not None:
        path = os.path.join(output_dir, f"chkpt_ep{epoch}.pyth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    checkpoints = glob.glob(os.path.join(output_dir, "chkpt_ep*.pyth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    epochs = [int(os.path.basename(c).split('ep')[1].split('.')[0]) for c in checkpoints]
    latest = max(epochs)
    print(f"Found latest checkpoint: epoch {latest}")
    return os.path.join(output_dir, f"chkpt_ep{latest}.pyth")


# ───────────────────────────────────────────────────────────────────────────
# Route-level statistics (reporting only)
# ───────────────────────────────────────────────────────────────────────────
def _pad_with_zeros(src_it):
    """Yield elements from the iterator, then infinite zeros."""
    yield from src_it
    yield from repeat(0)


def compute_late_sets_from_routes(nodes, routes, veh_speed):
    """
    Route-simulation statistics used for reporting:
      late_pickup : arrival at node later than (spoilage - direct return time)
      late_depot  : depot arrival after the node's spoilage time
      unserved    : feasible nodes never visited
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
            arrival_time = current_time + dist / veh_speed

            spoilage_time = nodes[node_idx, 3].item()
            direct_time_to_depot = torch.norm(depot_pos - node_pos).item() / veh_speed
            latest_pickup_time = spoilage_time - direct_time_to_depot

            if arrival_time > latest_pickup_time:
                late_pickup.add(node_idx)

            pickup_info.append((node_idx, spoilage_time))
            current_pos = node_pos
            current_time = arrival_time

        depot_arrival_time = current_time + torch.norm(depot_pos - current_pos).item() / veh_speed
        for node_idx, spoilage_time in pickup_info:
            if depot_arrival_time > spoilage_time:
                late_depot.add(node_idx)

    depot_pos_expand = nodes[0:1, :2]
    to_node_dist = torch.norm(nodes[:, :2] - depot_pos_expand, dim=-1)
    round_trip_time = 2 * (to_node_dist / veh_speed)
    infeasible_nodes = set(torch.nonzero(round_trip_time > nodes[:, 3]).flatten().tolist())
    infeasible_nodes.discard(0)

    unserved = (all_nodes - served_nodes) - infeasible_nodes
    return late_pickup, late_depot, unserved


def extract_learned_routes_from_actions(actions, batch_idx, veh_count):
    """
    Per-instance routes from learner actions.
    Empty routes are kept so the list always has veh_count entries.
    """
    instance_routes = [[] for _ in range(veh_count)]
    for veh_idx, node_idx in actions:
        v = veh_idx[batch_idx].item()
        n = node_idx[batch_idx].item()
        if n > 0:
            instance_routes[v].append(n)
    return instance_routes


# ───────────────────────────────────────────────────────────────────────────
# Cost of a-priori routes, computed inside the environment
# ───────────────────────────────────────────────────────────────────────────
def cost_via_env_for_apriori_routes(data, all_routes, batch_indices, device):
    """
    Evaluate fixed routes by stepping PVRP_Environment, so costs match the
    training objective exactly. Route lists shorter than veh_count are padded
    with empty routes, since a solver may leave a vehicle idle.
    """
    batch_nodes = data.nodes[batch_indices].to(device)
    batch_mask = (data.cust_mask[batch_indices].to(device)
                  if getattr(data, 'cust_mask', None) is not None else None)

    batch_data = PVRP_Dataset(
        nodes=batch_nodes,
        cust_mask=batch_mask,
        veh_count=data.veh_count,
        veh_capa=data.veh_capa,
        veh_speed=data.veh_speed,
    )

    env = _make_env(batch_data)
    env.reset()

    veh_count = data.veh_count
    batch_routes = [all_routes[i] for i in batch_indices.tolist()]

    routes_it = []
    for inst_routes in batch_routes:
        iters = [_pad_with_zeros(route) for route in inst_routes]
        while len(iters) < veh_count:
            iters.append(_pad_with_zeros([]))
        routes_it.append(iters)

    rewards = []
    while not env.done:
        cust_idx = env.nodes.new_tensor(
            [[next(routes_it[n][i.item()])] for n, i in enumerate(env.cur_veh_idx)],
            dtype=torch.int64,
        )
        rewards.append(env.step(cust_idx))

    return -torch.stack(rewards).sum(dim=0).squeeze(-1).detach().cpu()


# ───────────────────────────────────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────────────────────────────────
def evaluate_baseline_with_dataloader(data, routes, method_name):
    """OR-Tools / GA: costs from the environment, violations from route simulation."""
    stats = {'late_pickup': 0, 'late_delivery': 0, 'unserved': 0, 'costs': []}

    print("\n" + "=" * 80)
    print(f"EVALUATING {method_name.upper()} SOLUTIONS...")
    print("=" * 80)

    idx_loader = DataLoader(range(data.nodes.size(0)), batch_size=BATCH_SIZE, shuffle=False)

    for batch_indices in tqdm(idx_loader, desc=method_name):
        batch_costs = cost_via_env_for_apriori_routes(data, routes, batch_indices, device=dev)

        for local_i, global_idx in enumerate(batch_indices.tolist()):
            nodes = data.nodes[global_idx]
            late_p, late_d, unserved = compute_late_sets_from_routes(
                nodes, routes[global_idx], data.veh_speed
            )
            stats['late_pickup'] += len(late_p)
            stats['late_delivery'] += len(late_d - late_p)
            stats['unserved'] += len(unserved)
            stats['costs'].append(batch_costs[local_i].item())

    return stats


def evaluate_greedy_with_dataloader(data, learner):
    """RL greedy decoding."""
    stats = {'late_pickup': 0, 'late_delivery': 0, 'unserved': 0, 'costs': []}

    print("\n" + "=" * 80)
    print("EVALUATING RL GREEDY (ZERO-SHOT)...")
    print("=" * 80)

    learner.greedy = True
    n_inst = data.nodes.size(0)
    idx_loader = DataLoader(range(n_inst), batch_size=BATCH_SIZE, shuffle=False)

    t0 = time.time()
    for batch_indices in tqdm(idx_loader, desc="Greedy"):
        batch_nodes = data.nodes[batch_indices].to(dev)
        batch_mask = (data.cust_mask[batch_indices].to(dev)
                      if getattr(data, 'cust_mask', None) is not None else None)

        batch_data = PVRP_Dataset(
            nodes=batch_nodes, cust_mask=batch_mask,
            veh_count=data.veh_count, veh_capa=data.veh_capa, veh_speed=data.veh_speed,
        )

        env = _make_env(batch_data)
        with torch.no_grad():
            actions, _, rewards = learner(env)

        costs_batch = torch.stack(rewards).sum(0).mul(-1).squeeze(1).detach().cpu()

        for local_idx, global_idx in enumerate(batch_indices.tolist()):
            nodes_cpu = data.nodes[global_idx]
            inst_routes = extract_learned_routes_from_actions(actions, local_idx, data.veh_count)
            late_p, late_d, unserved = compute_late_sets_from_routes(
                nodes_cpu, inst_routes, data.veh_speed
            )
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

    The instance is replicated ROLLOUTS times along the batch dimension so all
    candidates are drawn in one forward pass instead of ROLLOUTS separate calls
    with batch size 1. This matters considerably at larger problem scales.
    """
    nodes = data.nodes[instance_idx]

    rep_nodes = nodes.unsqueeze(0).expand(ROLLOUTS, -1, -1).contiguous().to(dev)
    if getattr(data, 'cust_mask', None) is not None:
        rep_mask = data.cust_mask[instance_idx].unsqueeze(0) \
                       .expand(ROLLOUTS, -1).contiguous().to(dev)
    else:
        rep_mask = None

    rep_data = PVRP_Dataset(
        nodes=rep_nodes, cust_mask=rep_mask,
        veh_count=data.veh_count, veh_capa=data.veh_capa, veh_speed=data.veh_speed,
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
    """RL best-of-N sampling."""
    stats = {'late_pickup': 0, 'late_delivery': 0, 'unserved': 0, 'costs': []}

    print("\n" + "=" * 80)
    print("EVALUATING RL SAMPLING (ZERO-SHOT)...")
    print("=" * 80)

    n_inst = data.nodes.size(0)

    t0 = time.time()
    for global_idx in tqdm(range(n_inst), desc="Sampling"):
        best_route, best_cost = evaluate_sampling_single_instance(data, learner, global_idx)

        nodes = data.nodes[global_idx]
        late_p, late_d, unserved = compute_late_sets_from_routes(
            nodes, best_route, data.veh_speed
        )
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
NAME_OF = {
    'ortools': 'OR-Tools',
    'ga': 'Genetic Algorithm',
    'greedy': 'RL Greedy (Zero-Shot)',
    'sampling': 'RL Sampling (Zero-Shot)',
}


def print_generalization_comparison_report(results, total_patients, num_instances,
                                           train_config, test_config):
    print("\n" + "=" * 100)
    print(" COMPREHENSIVE ZERO-SHOT GENERALIZATION EVALUATION ".center(100, "="))
    print("=" * 100)

    print(f"\nTraining configuration: {train_config['n']} patients, {train_config['m']} ambulances")
    print(f"Test configuration:     {test_config['n']} patients, {test_config['m']} ambulances")
    print(f"Scale factor:           {test_config['n']/train_config['n']:.1f}x patients, "
          f"{test_config['m']/train_config['m']:.1f}x ambulances")

    print(f"\nTest dataset:")
    print(f"  Instances:         {num_instances}")
    print(f"  Total patients:    {total_patients}")
    print(f"  Patients/instance: {total_patients // num_instances}")

    print("\nCost parameters (matching environment):")
    for key, value in COST_PARAMS.items():
        print(f"  {key:25s} = {value}")

    available = [m for m in ['ortools', 'ga', 'greedy', 'sampling'] if m in results]

    for method in available:
        d = results[method]
        avg_cost = sum(d['costs']) / len(d['costs'])
        std_cost = torch.tensor(d['costs']).std().item()
        total_viol = d['late_pickup'] + d['late_delivery'] + d['unserved']

        print(f"\n{'─'*100}")
        print(f" {NAME_OF[method]} ".center(100, "─"))
        print(f"{'─'*100}")
        print(f"  Average cost:        {avg_cost:7.2f} ± {std_cost:5.2f}")
        print(f"  Late pickups:        {d['late_pickup']:5d}  ({100*d['late_pickup']/total_patients:6.2f}%)")
        print(f"  Late deliveries:     {d['late_delivery']:5d}  ({100*d['late_delivery']/total_patients:6.2f}%)")
        print(f"  Unserved patients:   {d['unserved']:5d}  ({100*d['unserved']/total_patients:6.2f}%)")
        print(f"  Total violations:    {total_viol:5d}")

    # ── Summary table ──
    print("\n" + "=" * 100)
    print(" SUMMARY TABLE: ALL METHODS ".center(100, "="))
    print("=" * 100)
    print(f"\n{'Method':<26} {'Avg Cost':>12} {'Std':>10} {'Late P':>8} "
          f"{'Late D':>8} {'Unserved':>10} {'Total Viol':>12}")
    print("─" * 100)
    for method in available:
        d = results[method]
        avg_cost = sum(d['costs']) / len(d['costs'])
        std_cost = torch.tensor(d['costs']).std().item()
        total_viol = d['late_pickup'] + d['late_delivery'] + d['unserved']
        print(f"{NAME_OF[method]:<26} {avg_cost:>12.4f} {std_cost:>10.4f} "
              f"{d['late_pickup']:>8d} {d['late_delivery']:>8d} "
              f"{d['unserved']:>10d} {total_viol:>12d}")

    # ── Comparison against OR-Tools ──
    if 'ortools' in results:
        print("\n" + "=" * 100)
        print(" COMPARISON VS OR-TOOLS ".center(100, "="))
        print("=" * 100)

        ort = results['ortools']
        ort_avg = sum(ort['costs']) / len(ort['costs'])

        for method in [m for m in available if m != 'ortools']:
            d = results[method]
            avg = sum(d['costs']) / len(d['costs'])
            diff = avg - ort_avg
            pct = 100 * diff / ort_avg if ort_avg else 0.0

            dp = d['late_pickup'] - ort['late_pickup']
            dd = d['late_delivery'] - ort['late_delivery']
            du = d['unserved'] - ort['unserved']

            print(f"\n{NAME_OF[method]} vs OR-Tools:")
            print(f"  Cost difference:      {diff:+7.2f}  ({pct:+6.1f}%)")
            print(f"  Late pickup delta:    {dp:+5d}")
            print(f"  Late delivery delta:  {dd:+5d}")
            print(f"  Unserved delta:       {du:+5d}")
            print(f"  Total violation delta:{dp+dd+du:+5d}")

    # ── Timing ──
    print("\n" + "=" * 100)
    print(" TIMING SUMMARY ".center(100, "="))
    print("=" * 100)
    print(f"\n{'Method':<26} {'Total (s)':>12} {'Per instance (ms)':>20} {'Execution':>18}")
    print("─" * 100)
    exec_mode = {'ortools': 'parallel (Pool)', 'ga': 'parallel (Pool)',
                 'greedy': 'sequential', 'sampling': 'sequential'}
    for key in ['ortools', 'ga', 'greedy', 'sampling']:
        if key in TIMINGS:
            total_s, per_ms = TIMINGS[key]
            print(f"{NAME_OF[key]:<26} {total_s:>12.1f} {per_ms:>20.2f} {exec_mode[key]:>18}")

    print("\nNote: OR-Tools and GA are solved across all CPU cores via multiprocessing.Pool,")
    print("      while RL inference runs in a single process. Wall-clock times are therefore")
    print("      not directly comparable without accounting for this difference.")

    print("\n" + "=" * 100 + "\n")


# ───────────────────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────────────────
def evaluate_comprehensive_generalization(pretrained_model_path, test_data_path, output_dir,
                                          train_config, test_config,
                                          ga_population=100, ga_generations=500):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading test data from {test_data_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    data = torch.load(test_data_path)

    num_instances = data.nodes.size(0)
    num_customers = data.nodes.size(1) - 1
    total_patients = num_instances * num_customers

    print(f"Test data loaded: {num_instances} instances, {num_customers} patients, "
          f"{data.veh_count} ambulances")
    print(f"Total patients: {total_patients}")

    learner = AttentionLearner(
        cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
        veh_state_size=PVRP_Environment.VEH_STATE_SIZE,
    )

    print(f"\nLoading pre-trained model from {pretrained_model_path}")
    chkpt = torch.load(pretrained_model_path, map_location="cpu")
    if "model" in chkpt:
        load_old_weights(learner, chkpt["model"])
    elif "learner" in chkpt:
        learner.load_state_dict(chkpt["learner"])
    else:
        raise KeyError(f"Unknown checkpoint format. Keys: {list(chkpt.keys())}")

    learner.to(dev)
    learner.eval()
    print("Model loaded")

    print("\n" + "=" * 80)
    print(" GENERATING BASELINE SOLUTIONS...")
    print("=" * 80)

    # Classical solvers operate on unnormalized coordinates and times
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

    print(f"\n2. Solving with Genetic Algorithm...")
    print(f"   Parameters: population={ga_population}, generations={ga_generations}")
    ga_start = time.time()
    ga_routes = ga_solve(unnormed_data, population_size=ga_population, generations=ga_generations)
    ga_time = time.time() - ga_start
    TIMINGS['ga'] = (ga_time, 1000.0 * ga_time / num_instances)
    print(f"   GA complete in {ga_time:.2f}s ({ga_time/num_instances:.3f}s per instance)")

    data.normalize()

    print("\n3. Evaluating all methods...")
    results = {}
    results['ortools'] = evaluate_baseline_with_dataloader(data, ref_routes, "OR-Tools")
    results['ga'] = evaluate_baseline_with_dataloader(data, ga_routes, "GA")
    results['greedy'] = evaluate_greedy_with_dataloader(data, learner)
    results['sampling'] = evaluate_sampling_with_dataloader(data, learner)

    print_generalization_comparison_report(results, total_patients, num_instances,
                                           train_config, test_config)

    save_path = os.path.join(output_dir, "generalization_comprehensive.pyth")
    torch.save({'results': results, 'timings': TIMINGS}, save_path)

    summary_path = os.path.join(output_dir, "generalization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Comprehensive Zero-Shot Generalization Evaluation\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training: {train_config['n']} patients, {train_config['m']} ambulances\n")
        f.write(f"Testing:  {test_config['n']} patients, {test_config['m']} ambulances\n")
        f.write(f"Scale:    {test_config['n']/train_config['n']:.1f}x patients, "
                f"{test_config['m']/train_config['m']:.1f}x ambulances\n\n")
        f.write(f"Model: {pretrained_model_path}\n")
        f.write(f"Test data: {test_data_path}\n")
        f.write(f"Instances: {num_instances}, total patients: {total_patients}\n\n")

        for method in ['ortools', 'ga', 'greedy', 'sampling']:
            if method not in results:
                continue
            d = results[method]
            avg_cost = sum(d['costs']) / len(d['costs'])
            std_cost = torch.tensor(d['costs']).std().item()
            total_viol = d['late_pickup'] + d['late_delivery'] + d['unserved']
            f.write(f"\n{NAME_OF[method]}:\n")
            f.write(f"  Cost: {avg_cost:.4f} +- {std_cost:.4f}\n")
            f.write(f"  Late pickups: {d['late_pickup']} ({100*d['late_pickup']/total_patients:.2f}%)\n")
            f.write(f"  Late deliveries: {d['late_delivery']} ({100*d['late_delivery']/total_patients:.2f}%)\n")
            f.write(f"  Unserved: {d['unserved']} ({100*d['unserved']/total_patients:.2f}%)\n")
            f.write(f"  Total violations: {total_viol}\n")
            if method in TIMINGS:
                f.write(f"  Time per instance: {TIMINGS[method][1]:.2f} ms\n")

    print(f"\nResults saved to {output_dir}")
    return results


def main(args):
    # ── Configuration ──
    train_n = 10
    train_m = 2
    train_epoch = 1000
    train_date = "260905-1821"      # critic run; use 260905-1821 for the rollout variant

    test_n = 15
    test_m = 3

    train_config = {'n': train_n, 'm': train_m}
    test_config = {'n': test_n, 'm': test_m}

    print("=" * 80)
    print(" COMPREHENSIVE ZERO-SHOT GENERALIZATION EVALUATION ".center(80))
    print("=" * 80)
    print(f"Training: {train_n} patients, {train_m} ambulances")
    print(f"Testing:  {test_n} patients, {test_m} ambulances")
    print(f"Scale:    {test_n/train_n:.1f}x patients, {test_m/train_m:.1f}x ambulances")
    print("=" * 80)

    try:
        model_dir = find_model_directory_by_date(
            f"./output/PVRPn{train_n}m{train_m}_", date=train_date
        )
        pretrained_model = find_latest_checkpoint(model_dir, epoch=train_epoch)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return None

    test_data = (f"./data/pvrp_n{test_n}m{test_m}/"
                 f"norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth")

    output_dir = (f"./results/generalization_comprehensive_"
                  f"n{train_n}m{train_m}_to_n{test_n}m{test_m}_"
                  f"{time.strftime('%y%m%d-%H%M')}/")

    try:
        return evaluate_comprehensive_generalization(
            pretrained_model_path=pretrained_model,
            test_data_path=test_data,
            output_dir=output_dir,
            train_config=train_config,
            test_config=test_config,
            ga_population=getattr(args, 'ga_population', 100),
            ga_generations=getattr(args, 'ga_generations', 500),
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main(parse_args())