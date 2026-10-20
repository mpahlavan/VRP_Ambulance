
"""
Route comparison between the critic and rollout policies on the same instance.

"""

import os
import time
import numpy as np
import torch

from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.utils import *
from marpdan.dep import matplotlib as mpl, pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
CRITIC_DATE = "260908-1016"
ROLLOUT_DATE = "260905-1821"
EPOCH = 1000

ROLLOUTS = 100            # best-of-N, matching the paper
INFERENCE = "sampling"    # "sampling" or "greedy"
N_FIGURES = 6

COST_PARAMS = {
    'spoilage_penalty': 1,
    'additional_late_penalty': 1,
    'unserved_penalty': 2,
    'idle_penalty_coef': 0.5,
    'dist_penalty_coef': 0.05,
}

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.style.use('seaborn-v0_8')
mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

HOSPITAL_COLOR = '#C0392B'
# One color per ambulance, shared across both panels so routes can be matched
AMBULANCE_COLORS = ['#1B6CA8', '#D1495B', '#2E8B57', '#8E6C8A', '#E07A00']


def make_env(batch_data):
    return PVRP_Environment(
        batch_data,
        spoilage_penalty=COST_PARAMS['spoilage_penalty'],
        unserved_penalty=COST_PARAMS['unserved_penalty'],
        additional_late_penalty=COST_PARAMS['additional_late_penalty'],
        dist_penalty_coef=COST_PARAMS['dist_penalty_coef'],
        idle_penalty_coef=COST_PARAMS['idle_penalty_coef'],
        logger=None,
    )


def load_policy(n, m, date, epoch):
    path = f"./output/PVRPn{n}m{m}_{date}/chkpt_ep{epoch}.pyth"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    learner = AttentionLearner(
        cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
        veh_state_size=PVRP_Environment.VEH_STATE_SIZE,
    )
    chkpt = torch.load(path, map_location="cpu")
    if "model" in chkpt:
        load_old_weights(learner, chkpt["model"])
    elif "learner" in chkpt:
        learner.load_state_dict(chkpt["learner"])
    else:
        raise KeyError(f"Unknown checkpoint format: {list(chkpt.keys())}")

    learner.to(dev)
    learner.eval()
    return learner


# ── Route extraction and violation accounting ──────────────────────────────
def extract_routes(actions, batch_idx, veh_count):
    routes = [[] for _ in range(veh_count)]
    for veh_idx, node_idx in actions:
        v = veh_idx[batch_idx].item()
        n = node_idx[batch_idx].item()
        if n > 0:
            routes[v].append(n)
    return routes


def route_lengths(nodes, routes):
    """
    Euclidean length of each route, hospital to hospital, in problem units.
    Returns the per-ambulance lengths and their total.
    """
    depot = nodes[0, :2]
    lengths = []
    for route in routes:
        if not route:
            lengths.append(0.0)
            continue
        total = 0.0
        pos = depot
        for n in route:
            total += torch.norm(nodes[n, :2] - pos).item()
            pos = nodes[n, :2]
        total += torch.norm(depot - pos).item()
        lengths.append(total)
    return lengths, sum(lengths)


def compute_violations(nodes, routes, veh_speed):
    late_pickup, late_hospital = set(), set()
    depot_pos = nodes[0, :2]
    served = set()

    for route in routes:
        if not route:
            continue

        current_pos = depot_pos.clone()
        current_time = 0.0
        onboard = []

        for node_idx in route:
            served.add(node_idx)
            node_pos = nodes[node_idx, :2]
            arrival = current_time + torch.norm(node_pos - current_pos).item() / veh_speed

            spoilage = nodes[node_idx, 3].item()
            return_time = torch.norm(depot_pos - node_pos).item() / veh_speed
            if arrival > spoilage - return_time:
                late_pickup.add(node_idx)

            onboard.append((node_idx, spoilage))
            current_pos = node_pos
            current_time = arrival

        depot_arrival = current_time + torch.norm(depot_pos - current_pos).item() / veh_speed
        for node_idx, spoilage in onboard:
            if depot_arrival > spoilage:
                late_hospital.add(node_idx)

    to_depot = torch.norm(nodes[:, :2] - nodes[0:1, :2], dim=-1)
    infeasible = set(torch.nonzero(2 * to_depot / veh_speed > nodes[:, 3]).flatten().tolist())
    infeasible.discard(0)

    unserved = (set(range(1, len(nodes))) - served) - infeasible
    return late_pickup, late_hospital, unserved, infeasible


# ── Policy evaluation ──────────────────────────────────────────────────────
def run_policy(data, learner, mode):
    n_inst = data.nodes.size(0)
    veh_count = data.veh_count
    costs = torch.zeros(n_inst)
    all_routes = [None] * n_inst

    if mode == "greedy":
        learner.greedy = True
        batch = 32
        for start in range(0, n_inst, batch):
            idx = list(range(start, min(start + batch, n_inst)))
            batch_data = PVRP_Dataset(
                nodes=data.nodes[idx].to(dev),
                cust_mask=data.cust_mask[idx].to(dev) if data.cust_mask is not None else None,
                veh_count=veh_count, veh_capa=data.veh_capa, veh_speed=data.veh_speed,
            )
            env = make_env(batch_data)
            with torch.no_grad():
                actions, _, rewards = learner(env)
            batch_costs = torch.stack(rewards).sum(0).mul(-1).squeeze(1).detach().cpu()
            for local, g in enumerate(idx):
                costs[g] = batch_costs[local]
                all_routes[g] = extract_routes(actions, local, veh_count)
    else:
        learner.greedy = False
        for g in range(n_inst):
            rep_nodes = data.nodes[g].unsqueeze(0).expand(ROLLOUTS, -1, -1).contiguous().to(dev)
            rep_mask = (data.cust_mask[g].unsqueeze(0).expand(ROLLOUTS, -1).contiguous().to(dev)
                        if data.cust_mask is not None else None)
            rep_data = PVRP_Dataset(
                nodes=rep_nodes, cust_mask=rep_mask,
                veh_count=veh_count, veh_capa=data.veh_capa, veh_speed=data.veh_speed,
            )
            env = make_env(rep_data)
            with torch.no_grad():
                actions, _, rewards = learner(env)
            sample_costs = torch.stack(rewards).sum(0).mul(-1).squeeze(-1).detach().cpu()
            best = int(sample_costs.argmin().item())
            costs[g] = sample_costs[best]
            all_routes[g] = extract_routes(actions, best, veh_count)

    return costs, all_routes


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_instance(ax, nodes, routes, lengths, title, late_pickup, late_hospital,
                  unserved, infeasible):
    ax.plot(nodes[0, 0].item(), nodes[0, 1].item(),
            marker='H', markersize=16, color=HOSPITAL_COLOR,
            markeredgecolor='black', markeredgewidth=1.5,
            linestyle='None', label='Hospital', zorder=10)

    scatter = ax.scatter(nodes[1:, 0], nodes[1:, 1],
                         c=nodes[1:, 3], cmap='RdYlGn',
                         edgecolors='k', linewidths=0.4, s=150,
                         label='Patient (color: survival time)', zorder=5)
    plt.colorbar(scatter, ax=ax, shrink=0.75, label='Survival time')

    # Color identifies the ambulance, and is the same in both panels, so that
    # a route reassigned between ambulances is visible as a color change.
    for r_idx, route in enumerate(routes):
        if not route:
            continue
        color = AMBULANCE_COLORS[r_idx % len(AMBULANCE_COLORS)]
        pts = nodes[[0] + route + [0]]
        ax.plot(pts[:, 0], pts[:, 1], '-', color=color,
                alpha=0.85, linewidth=2.5,
                label=f'Ambulance {r_idx + 1} (length {lengths[r_idx]:.3f})',
                zorder=3)
        for i in range(len(pts) - 1):
            ax.annotate('', xy=(pts[i + 1, 0], pts[i + 1, 1]),
                        xytext=(pts[i, 0], pts[i, 1]),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.8, alpha=0.7))

    hospital_only = late_hospital - late_pickup

    for n in sorted(infeasible):
        ax.scatter(nodes[n, 0], nodes[n, 1], marker='^', s=230,
                   facecolors='none', edgecolors='dimgray', linewidths=2.0,
                   label='Structurally infeasible' if n == min(infeasible) else '',
                   zorder=6)
    for n in sorted(late_pickup):
        ax.scatter(nodes[n, 0], nodes[n, 1], marker='o', s=210,
                   facecolors='none', edgecolors='red', linewidths=2.6,
                   label='Late pickup' if n == min(late_pickup) else '',
                   zorder=7)
    for n in sorted(hospital_only):
        ax.scatter(nodes[n, 0], nodes[n, 1], marker='o', s=230,
                   facecolors='none', edgecolors='darkorange', linewidths=2.4,
                   label='Late at hospital' if n == min(hospital_only) else '',
                   zorder=7)
    for n in sorted(unserved):
        ax.plot(nodes[n, 0], nodes[n, 1], marker='x', markersize=12,
                color='black', markeredgewidth=2.0, linestyle='None',
                label='Unserved' if n == min(unserved) else '', zorder=8)

    ax.set_title(title, fontsize=12, pad=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)


def summarize(cost, total_length, late_p, late_h, unserved, served_count):
    return (f"cost {cost:.3f} | total route length {total_length:.3f} | "
            f"served {served_count}\n"
            f"late pickup {len(late_p)} | late hospital {len(late_h - late_p)} | "
            f"unserved {len(unserved)}")


def main(args):
    n, m = args.customers_count, args.vehicles_count
    data_path = (f"./data/pvrp_n{n}m{m}/"
                 f"norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth")

    print(f"Loading test data from {data_path}")
    data = torch.load(data_path)
    n_inst = data.nodes.size(0)
    print(f"{n_inst} instances, {n} patients, {data.veh_count} ambulances")

    print(f"\nLoading critic policy   ({CRITIC_DATE})")
    critic = load_policy(n, m, CRITIC_DATE, EPOCH)
    print(f"Loading rollout policy  ({ROLLOUT_DATE})")
    rollout = load_policy(n, m, ROLLOUT_DATE, EPOCH)

    print(f"\nRunning both policies in {INFERENCE} mode...")
    t0 = time.time()
    critic_costs, critic_routes = run_policy(data, critic, INFERENCE)
    rollout_costs, rollout_routes = run_policy(data, rollout, INFERENCE)
    print(f"done in {time.time() - t0:.1f}s")
    print(f"  critic  mean cost {critic_costs.mean():.4f}")
    print(f"  rollout mean cost {rollout_costs.mean():.4f}")

    # ── Per-instance violation counts for both policies ──
    rows = []
    for idx in range(n_inst):
        nodes = data.nodes[idx].cpu()
        c_routes = [r for r in critic_routes[idx] if r]
        r_routes = [r for r in rollout_routes[idx] if r]

        c_lp, c_lh, c_un, infeasible = compute_violations(nodes, c_routes, data.veh_speed)
        r_lp, r_lh, r_un, _ = compute_violations(nodes, r_routes, data.veh_speed)

        _, c_len = route_lengths(nodes, c_routes)
        _, r_len = route_lengths(nodes, r_routes)

        rows.append({
            'idx': idx,
            'c_cost': critic_costs[idx].item(),
            'r_cost': rollout_costs[idx].item(),
            'c_temporal': len(c_lp) + len(c_lh - c_lp),
            'r_temporal': len(r_lp) + len(r_lh - r_lp),
            'c_unserved': len(c_un),
            'r_unserved': len(r_un),
            'c_length': c_len,
            'r_length': r_len,
        })

    # ── How often do the two policies actually differ? ──
    cost_diff = np.array([r['c_cost'] - r['r_cost'] for r in rows])
    identical = np.isclose(cost_diff, 0.0, atol=1e-6)
    temporal_diff = np.array([r['c_temporal'] - r['r_temporal'] for r in rows])
    unserved_diff = np.array([r['c_unserved'] - r['r_unserved'] for r in rows])
    length_diff = np.array([r['c_length'] - r['r_length'] for r in rows])

    print("\n" + "=" * 72)
    print("DISTRIBUTION OF PER-INSTANCE DIFFERENCES (critic minus rollout)")
    print("=" * 72)
    print(f"  identical cost:            {identical.sum():4d} / {n_inst}  "
          f"({100*identical.mean():.1f}%)")
    print(f"  critic cheaper:            {(cost_diff < -1e-6).sum():4d} "
          f"({100*(cost_diff < -1e-6).mean():.1f}%)")
    print(f"  rollout cheaper:           {(cost_diff >  1e-6).sum():4d} "
          f"({100*(cost_diff >  1e-6).mean():.1f}%)")
    print(f"  mean cost difference:      {cost_diff.mean():+.4f}")
    print(f"  median cost difference:    {np.median(cost_diff):+.4f}")
    print(f"  mean over differing only:  {cost_diff[~identical].mean():+.4f}"
          if (~identical).any() else "  (no differing instances)")
    print()
    print(f"  mean route-length difference:   {length_diff.mean():+.4f}")
    print(f"  median route-length difference: {np.median(length_diff):+.4f}")
    print(f"  implied mean cost from distance "
          f"({COST_PARAMS['dist_penalty_coef']} per unit): "
          f"{COST_PARAMS['dist_penalty_coef'] * length_diff.mean():+.4f}")
    print()
    print(f"  critic has more temporal violations: "
          f"{(temporal_diff > 0).sum():4d} ({100*(temporal_diff > 0).mean():.1f}%)")
    print(f"  critic serves more patients:         "
          f"{(unserved_diff < 0).sum():4d} ({100*(unserved_diff < 0).mean():.1f}%)")
    print(f"  both at once (characteristic trade): "
          f"{((temporal_diff > 0) & (unserved_diff < 0)).sum():4d} "
          f"({100*((temporal_diff > 0) & (unserved_diff < 0)).mean():.1f}%)")

    # ── Instance selection ──
    # Rule: among instances exhibiting the characteristic trade-off, that is
    # the critic policy incurring more temporal violations while leaving fewer
    # patients unserved, take those whose cost difference is closest to the
    # mean difference over that subset. This targets the behavior the paper
    # reports, and the rule is applied to counts rather than to the figures.
    candidates = [i for i in range(n_inst)
                  if temporal_diff[i] > 0 and unserved_diff[i] < 0]

    if not candidates:
        print("\nNo instance shows the characteristic trade-off; "
              "falling back to the largest temporal-violation difference.")
        candidates = list(np.argsort(-temporal_diff)[:N_FIGURES])
        selected = candidates[:N_FIGURES]
        rule = ("instances with the largest difference in temporal violations "
                "between the two policies")
    else:
        sub = cost_diff[candidates]
        target = sub.mean()
        order = np.argsort(np.abs(sub - target))
        selected = [candidates[i] for i in order[:N_FIGURES]]
        rule = (f"instances in which the critic policy incurs more temporal "
                f"violations while leaving fewer patients unserved "
                f"({len(candidates)} of {n_inst} instances), with cost "
                f"difference closest to the mean over that subset")

    print(f"\nSelection rule: {rule}")
    print(f"Selected instances: {selected}")

    outdir = (f"results/critic_vs_rollout_n{n}m{m}_{INFERENCE}_"
              f"{time.strftime('%y%m%d-%H%M')}")
    os.makedirs(outdir, exist_ok=True)

    for idx in selected:
        nodes = data.nodes[idx].cpu()
        c_routes = [r for r in critic_routes[idx] if r]
        r_routes = [r for r in rollout_routes[idx] if r]

        c_lp, c_lh, c_un, infeasible = compute_violations(nodes, c_routes, data.veh_speed)
        r_lp, r_lh, r_un, _ = compute_violations(nodes, r_routes, data.veh_speed)

        c_lengths, c_total = route_lengths(nodes, c_routes)
        r_lengths, r_total = route_lengths(nodes, r_routes)

        c_served = sum(len(r) for r in c_routes)
        r_served = sum(len(r) for r in r_routes)

        fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(20, 9.5))

        plot_instance(ax_c, nodes, c_routes, c_lengths,
                      "Learned critic baseline\n" +
                      summarize(critic_costs[idx].item(), c_total,
                                c_lp, c_lh, c_un, c_served),
                      c_lp, c_lh, c_un, infeasible)

        plot_instance(ax_r, nodes, r_routes, r_lengths,
                      "Monte Carlo rollout baseline\n" +
                      summarize(rollout_costs[idx].item(), r_total,
                                r_lp, r_lh, r_un, r_served),
                      r_lp, r_lh, r_un, infeasible)

        fig.suptitle(f"Test instance {idx} — {INFERENCE} inference", fontsize=14)
        fig.tight_layout()

        path = f"{outdir}/instance_{idx:04d}.pdf"
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"  wrote {path}  "
              f"(cost {critic_costs[idx]:.3f} vs {rollout_costs[idx]:.3f}, "
              f"length {c_total:.3f} vs {r_total:.3f})")

    # ── Statistics and caption, so the selection rule travels with the figure ──
    with open(f"{outdir}/statistics.txt", "w") as f:
        f.write(f"Per-instance comparison, {INFERENCE} inference, "
                f"{n_inst} test instances\n")
        f.write("=" * 72 + "\n")
        f.write(f"identical cost:   {identical.sum()} ({100*identical.mean():.1f}%)\n")
        f.write(f"critic cheaper:   {(cost_diff < -1e-6).sum()}\n")
        f.write(f"rollout cheaper:  {(cost_diff > 1e-6).sum()}\n")
        f.write(f"mean cost difference:     {cost_diff.mean():+.4f}\n")
        f.write(f"median cost difference:   {np.median(cost_diff):+.4f}\n")
        f.write(f"mean length difference:   {length_diff.mean():+.4f}\n")
        f.write(f"median length difference: {np.median(length_diff):+.4f}\n")
        f.write(f"characteristic trade-off present in "
                f"{((temporal_diff > 0) & (unserved_diff < 0)).sum()} instances\n")

    with open(f"{outdir}/caption.txt", "w") as f:
        f.write(
            f"Routing solutions produced by the two policy variants on the same test "
            f"instance under {INFERENCE} inference. Line color identifies the ambulance "
            f"and is consistent across panels. Instances were selected as {rule}. "
            f"Red circles mark late pickups, orange circles late hospital arrivals, "
            f"crosses unserved patients, and triangles patients whose direct round trip "
            f"from the hospital already exceeds their survival time. Patient color "
            f"encodes survival time. Total route length is given in problem units; "
            f"with a distance coefficient of {COST_PARAMS['dist_penalty_coef']}, "
            f"differences in length of the magnitude seen here contribute little to "
            f"total cost relative to the deadline and coverage penalties.\n"
        )
    print(f"\nStatistics written to {outdir}/statistics.txt")
    print(f"Caption written to {outdir}/caption.txt")


if __name__ == "__main__":
    main(parse_args())