#!/usr/bin/env python3
"""
Visualize routes from Genetic Algorithm solver
Compares GA routes with OR-Tools on selected instances
"""

import os
import time
import torch
import numpy as np
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ga_solve, ort_solve
from marpdan.utils import parse_args, eval_apriori_routes
from marpdan.dep import matplotlib as mpl, pyplot as plt

plt.style.use('seaborn-v0_8')
mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.titlesize': 15,
    'legend.fontsize': 10,
})

COLORS = {
    'ortools': '#2E86AB',
    'ga': '#28A745',
    'hospital': '#FF6B6B',
    'late_pickup': '#FF3838',
    'late_depot': '#FF6348',
    'unserved': '#8B0000',
    'infeasible': '#696969'
}


def compute_late_sets_from_routes(nodes, routes, veh_speed):
    """
    Compute late sets by simulating routes directly.

    Args:
        nodes: Tensor of shape [num_nodes, 4] with (x, y, demand, spoilage_time)
        routes: List of routes, each route is a list of node indices
        veh_speed: Vehicle speed

    Returns:
        late_pickup: set of nodes with late pickup
        late_depot: set of nodes with late hospital delivery
    """
    late_pickup = set()
    late_depot = set()

    depot_pos = nodes[0, :2]

    for route in routes:
        if not route:
            continue

        current_pos = depot_pos.clone()
        current_time = 0.0
        nodes_on_vehicle = []

        for node_idx in route:
            node_pos = nodes[node_idx, :2]
            dist = torch.norm(node_pos - current_pos).item()
            travel_time = dist / veh_speed
            arrival_time = current_time + travel_time

            spoilage_time = nodes[node_idx, 3].item()
            dist_to_depot = torch.norm(depot_pos - node_pos).item()
            return_time = dist_to_depot / veh_speed
            latest_pickup_time = spoilage_time - return_time

            if arrival_time > latest_pickup_time:
                late_pickup.add(node_idx)

            nodes_on_vehicle.append((node_idx, spoilage_time))
            current_pos = node_pos
            current_time = arrival_time

        # Return to depot
        dist_to_depot = torch.norm(depot_pos - current_pos).item()
        depot_arrival_time = current_time + dist_to_depot / veh_speed

        for node_idx, spoilage_time in nodes_on_vehicle:
            if depot_arrival_time > spoilage_time:
                late_depot.add(node_idx)

    return late_pickup, late_depot


def compute_infeasible_nodes(nodes, veh_speed):
    """Compute nodes that cannot be served due to spoilage constraints"""
    depot_pos = nodes[0:1, :2]
    node_pos = nodes[:, :2]
    to_node_dist = torch.norm(node_pos - depot_pos, dim=-1)
    round_trip_time = 2 * (to_node_dist / veh_speed)
    infeasible_mask = round_trip_time > nodes[:, 3]
    infeasible_nodes = set(torch.nonzero(infeasible_mask).flatten().tolist())
    infeasible_nodes.discard(0)
    return infeasible_nodes


def plot_routes(ax, nodes, routes, title, late_nodes=None, late_depot_nodes=None,
                unserved_nodes=None, infeasible_nodes=None, route_color='#2E86AB'):
    """Plot PVRP routes on given axes"""
    late_nodes = late_nodes or set()
    late_depot_nodes = late_depot_nodes or set()
    unserved_nodes = unserved_nodes or set()
    infeasible_nodes = infeasible_nodes or set()

    # Hospital (depot)
    ax.plot(nodes[0, 0].item(), nodes[0, 1].item(),
            marker='H', markersize=15, color=COLORS['hospital'],
            markeredgecolor='black', markeredgewidth=2,
            label='Hospital', zorder=10)

    # Patients (nodes) - color by spoilage time
    if len(nodes) > 1:
        scatter = ax.scatter(nodes[1:, 0], nodes[1:, 1],
                            c=nodes[1:, 3],
                            cmap='RdYlGn',
                            edgecolors="k", linewidths=0.4,
                            s=180,
                            label='Patient (color=survival time)')
        plt.colorbar(scatter, ax=ax, shrink=0.8)

    # Routes
    route_styles = ['-', '--', '-.', ':']
    route_colors = plt.cm.Set1(np.linspace(0, 1, max(len(routes), 1)))

    for route_idx, (route, color) in enumerate(zip(routes, route_colors)):
        if not route:
            continue

        full_route = [0] + list(route) + [0]
        points = nodes[full_route]

        style = route_styles[route_idx % len(route_styles)]
        ax.plot(points[:, 0], points[:, 1], style,
                color=color, alpha=0.8, linewidth=3,
                label=f'Vehicle {route_idx + 1}')

        # Arrows
        for i in range(len(points) - 1):
            ax.annotate('', xy=(points[i + 1, 0], points[i + 1, 1]),
                       xytext=(points[i, 0], points[i, 1]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))

    late_depot_only = late_depot_nodes - late_nodes

    # Infeasible nodes
    if infeasible_nodes:
        for n in infeasible_nodes:
            ax.scatter(nodes[n, 0], nodes[n, 1],
                      marker='^', s=250, facecolors='none',
                      edgecolors='red', linewidths=2.5,
                      label="Infeasible" if n == min(infeasible_nodes) else "",
                      zorder=6)

    # Late pickup
    if late_nodes:
        for n in late_nodes:
            ax.scatter(nodes[n, 0], nodes[n, 1],
                      marker='o', s=220, facecolors='none',
                      edgecolors='red', linewidths=3.0,
                      label="Late pickup" if n == min(late_nodes) else "",
                      zorder=7)

    # Late at hospital
    if late_depot_only:
        for n in late_depot_only:
            ax.scatter(nodes[n, 0], nodes[n, 1],
                      marker='o', s=240, facecolors='none',
                      edgecolors='darkorange', linewidths=2.8,
                      label="Late at hospital" if n == min(late_depot_only) else "",
                      zorder=7)

    # Unserved
    if unserved_nodes:
        for n in unserved_nodes:
            ax.plot(nodes[n, 0], nodes[n, 1],
                   marker='x', markersize=12, color='black',
                   markeredgewidth=1.1, alpha=0.8,
                   label="Unserved" if n == min(unserved_nodes) else "",
                   zorder=8)

    ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect("equal")

    legend = ax.legend(loc="upper left", fontsize=9, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')


def visualize_ga_vs_ortools(data, ga_routes, ort_routes, ga_costs, ort_costs,
                            indices, output_dir, veh_speed):
    """
    Visualize GA vs OR-Tools routes for selected instances

    Args:
        data: PVRP_Dataset (normalized)
        ga_routes: List of GA routes per instance
        ort_routes: List of OR-Tools routes per instance
        ga_costs: Tensor of GA costs
        ort_costs: Tensor of OR-Tools costs
        indices: Instance indices to visualize
        output_dir: Output directory
        veh_speed: Vehicle speed
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        nodes = data.nodes[idx]

        ga_route = ga_routes[idx]
        ort_route = ort_routes[idx]

        ga_cost = ga_costs[idx].item()
        ort_cost = ort_costs[idx].item()
        gap = (ga_cost - ort_cost) / ort_cost * 100

        # Compute infeasible nodes
        infeasible = compute_infeasible_nodes(nodes, veh_speed)

        # Compute late sets
        late_ga, late_depot_ga = compute_late_sets_from_routes(nodes, ga_route, veh_speed)
        late_ort, late_depot_ort = compute_late_sets_from_routes(nodes, ort_route, veh_speed)

        # Compute served/unserved
        all_nodes = set(range(1, len(nodes)))
        ga_served = set([n for route in ga_route for n in route])
        ort_served = set([n for route in ort_route for n in route])

        ga_unserved = (all_nodes - ga_served) - infeasible
        ort_unserved = (all_nodes - ort_served) - infeasible

        # Create figure
        fig, (ax_ort, ax_ga) = plt.subplots(1, 2, figsize=(24, 10))

        # OR-Tools plot
        ort_title = (f"OR-Tools (Cost: {ort_cost:.2f})\n"
                    f"Served: {len(ort_served)} | Late Pickup: {len(late_ort)} | "
                    f"Late Hospital: {len(late_depot_ort - late_ort)} | Unserved: {len(ort_unserved)}")

        plot_routes(ax_ort, nodes, ort_route, ort_title,
                   late_nodes=late_ort, late_depot_nodes=late_depot_ort,
                   unserved_nodes=ort_unserved, infeasible_nodes=infeasible)

        # GA plot
        ga_title = (f"Genetic Algorithm (Cost: {ga_cost:.2f}, Gap: {gap:+.1f}%)\n"
                   f"Served: {len(ga_served)} | Late Pickup: {len(late_ga)} | "
                   f"Late Hospital: {len(late_depot_ga - late_ga)} | Unserved: {len(ga_unserved)}")

        plot_routes(ax_ga, nodes, ga_route, ga_title,
                   late_nodes=late_ga, late_depot_nodes=late_depot_ga,
                   unserved_nodes=ga_unserved, infeasible_nodes=infeasible)

        fig.suptitle(f"Instance {idx}: GA vs OR-Tools Comparison", fontsize=16, fontweight='bold')
        fig.tight_layout()

        filepath = f"{output_dir}/ga_vs_ort_instance_{idx:03d}_gap_{gap:+.0f}.pdf"
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"  Saved: {filepath}")


def main(args):
    pb = args.problem_type
    n = args.customers_count
    m = args.vehicles_count

    output_dir = f"./results/{pb}_n{n}m{m}_ga_plots_{time.strftime('%y%m%d-%H%M')}/"
    data_path = f"./data/{pb}_n{n}m{m}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"

    print(f"Loading data from: {data_path}")
    data = torch.load(data_path)

    # Create unnormalized dataset for solvers
    nodes_unnorm = data.nodes.clone()
    nodes_unnorm[:, :, :2] *= 100
    nodes_unnorm[:, :, 2] *= 200
    nodes_unnorm[:, :, 3] *= 480

    unnormed_data = PVRP_Dataset(data.veh_count, data.veh_capa, data.veh_speed, nodes_unnorm)

    # Environment for cost computation
    env = PVRP_Environment(data)

    # Solve with GA
    print("\nRunning Genetic Algorithm...")
    ga_routes = ga_solve(unnormed_data, population_size=100, generations=500)
    ga_costs = eval_apriori_routes(env, ga_routes, 1)

    # Solve with OR-Tools
    print("\nRunning OR-Tools...")
    ort_routes = ort_solve(unnormed_data)
    ort_costs = eval_apriori_routes(env, ort_routes, 1)

    # Select instances to visualize
    # Best GA (lowest gap), worst GA (highest gap), and middle
    gaps = ga_costs - ort_costs
    sorted_idx = gaps.argsort()

    n_samples = min(10, len(gaps))
    step = len(gaps) // n_samples

    # Select diverse instances: best, worst, and evenly spaced
    indices = []
    indices.extend(sorted_idx[:3].tolist())  # 3 best GA
    indices.extend(sorted_idx[-3:].tolist())  # 3 worst GA
    indices.extend(sorted_idx[::step][:4].tolist())  # 4 evenly spaced
    indices = list(set(indices))[:10]  # Unique, max 10

    print(f"\nVisualizing {len(indices)} instances...")

    # Visualize
    visualize_ga_vs_ortools(
        data=data,
        ga_routes=ga_routes,
        ort_routes=ort_routes,
        ga_costs=ga_costs,
        ort_costs=ort_costs,
        indices=indices,
        output_dir=output_dir,
        veh_speed=data.veh_speed
    )

    print(f"\nVisualization complete! Results saved to: {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GA Mean Cost: {ga_costs.mean():.4f}")
    print(f"OR-Tools Mean Cost: {ort_costs.mean():.4f}")
    print(f"Gap: {(ga_costs.mean() - ort_costs.mean()) / ort_costs.mean() * 100:+.2f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
