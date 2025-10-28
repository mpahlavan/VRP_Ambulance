#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate TikZ/LaTeX visualizations for PVRP routes.
Adapted from original VRP routes_to_tex.py for ambulance routing.
"""

from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import eval_apriori_routes, load_old_weights, parse_args
from itertools import cycle, tee
import torch
import subprocess
import os

# TikZ template for PVRP visualization
TIKZ_TMPL = r"""\documentclass[tikz, crop]{{standalone}}
\usetikzlibrary{{shapes,positioning,backgrounds}}
\begin{{document}}
\begin{{tikzpicture}}[x=3cm, y=3cm,
    depot/.style = {{draw, fill=red!20, minimum height=6mm, diamond, aspect=0.8, font=\footnotesize}},
    patient/.style = {{draw, fill=blue!20, circle, font=\scriptsize}},
    critical/.style = {{draw, fill=red!40, circle, font=\scriptsize}},
    late/.style = {{draw, fill=orange!60, circle, font=\scriptsize}},
    unserved/.style = {{draw, fill=gray!40, circle, font=\scriptsize}},
    spoilage/.style = {{font=\tiny, align=center}},
    every path/.append style = {{->, >=latex, line width=1.2pt}}]

\node[depot] (n0) at ({},{}) {{Hospital}};
{}
\begin{{pgfonlayer}}{{background}}
{}
\end{{pgfonlayer}}
\end{{tikzpicture}}
\end{{document}}"""

# Color cycle for different ambulances
COLORS = cycle(("blue!70", "green!60!black", "red!70", "orange!80", "purple!60", "brown!60"))

def prv_nxt(iterable):
    """Get previous-next pairs from iterable"""
    prv, nxt = tee(iterable, 2)
    yield (0, next(nxt))
    yield from zip(prv, nxt)

def tikz_patient_node(node, idx, spoilage_time, is_critical=False, is_late=False, is_unserved=False):
    """Generate TikZ node for a patient with spoilage time and status indicators"""
    
    # Determine node style based on status
    if is_unserved:
        style = "unserved"
        label = f"\\textbf{{{idx}}}"
    elif is_late:
        style = "late" 
        label = f"\\textbf{{{idx}}}"
    elif is_critical:
        style = "critical"
        label = f"\\textbf{{{idx}}}"
    else:
        style = "patient"
        label = f"{idx}"
    
    # Scale coordinates and spoilage time for better visualization
    x, y = node[0]/10, node[1]/10
    
    # Format spoilage time (assuming normalized values)
    spoil_display = f"{spoilage_time:.2f}"
    
    return f"""\\node[{style}, minimum size=5mm] (n{idx}) at ({x:.1f}, {y:.1f}) {{{label}}};
\\node[spoilage, above = 1mm of n{idx}] {{T={spoil_display}}};"""

def tikz_route(route, color, vehicle_id, route_info=None):
    """Generate TikZ path for an ambulance route with additional info"""
    if not route:
        return ""
    
    route_lines = []
    route_lines.append(f"% Ambulance {vehicle_id}")
    
    # Add route information as comment if provided
    if route_info:
        route_lines.append(f"% {route_info}")
    
    # Generate path commands
    path_commands = []
    for prv, nxt in prv_nxt(route):
        if nxt == 0:  # Return to hospital
            path_commands.append(f"\\draw[{color}, dashed] (n{prv}) -- (n{nxt});")
        else:
            path_commands.append(f"\\draw[{color}] (n{prv}) -- (n{nxt});")
    
    route_lines.extend(path_commands)
    return "\n".join(route_lines)

def analyze_route_quality(env, routes):
    """Analyze route quality and extract metrics"""
    metrics = {
        'total_served': 0,
        'total_late_pickup': 0, 
        'total_late_depot': 0,
        'total_unserved': 0,
        'vehicle_utilization': 0,
        'total_distance': 0.0
    }
    
    # Extract late nodes information
    late_pickup_nodes = set()
    if hasattr(env, 'late_nodes') and env.late_nodes is not None:
        late_pickup_indices = torch.nonzero(env.late_nodes[0], as_tuple=False)
        late_pickup_nodes = set(idx.item() for idx in late_pickup_indices.flatten() if idx.item() > 0)
    
    # Calculate late depot deliveries
    late_depot_nodes = set()
    if hasattr(env, 'vehicles') and hasattr(env, 'vehicle_routes'):
        arrival_times = env.vehicles[0, :, 3]  # Vehicle arrival times at depot
        spoilage_times = env.nodes[0, :, 3]    # Node spoilage times
        
        for v in range(env.veh_count):
            # Find nodes served by this vehicle
            served_by_v = torch.nonzero(env.vehicle_routes[0] == v, as_tuple=False)
            if served_by_v.numel() > 0:
                depot_arrival = arrival_times[v].item()
                for node_idx in served_by_v.flatten():
                    node = node_idx.item()
                    if node > 0 and depot_arrival > spoilage_times[node].item():
                        late_depot_nodes.add(node)
    
    # Count served nodes
    served_nodes = set()
    for route in routes:
        served_nodes.update(route)
    
    # Calculate metrics
    total_nodes = env.nodes.size(1) - 1  # Exclude depot
    metrics['total_served'] = len(served_nodes)
    metrics['total_late_pickup'] = len(late_pickup_nodes)
    metrics['total_late_depot'] = len(late_depot_nodes)
    metrics['total_unserved'] = total_nodes - len(served_nodes)
    metrics['vehicle_utilization'] = len([r for r in routes if r]) / env.veh_count * 100
    
    return metrics, late_pickup_nodes, late_depot_nodes

def generate_route_tikz(nodes, routes, method_name, metrics, late_pickup_nodes, late_depot_nodes, unserved_nodes):
    """Generate complete TikZ code for routes visualization"""
    
    # Generate patient nodes
    patient_nodes = []
    for idx in range(1, len(nodes)):  # Skip depot (index 0)
        node = nodes[idx]
        spoilage_time = node[3].item()  # Spoilage time
        
        # Determine node status
        is_critical = spoilage_time < 0.7  # Assuming normalized values, <0.7 is critical
        is_late = idx in late_pickup_nodes
        is_unserved = idx in unserved_nodes
        
        patient_nodes.append(tikz_patient_node(
            node, idx, spoilage_time, is_critical, is_late, is_unserved
        ))
    
    # Generate route paths
    route_paths = []
    colors = list(COLORS)
    for i, route in enumerate(routes):
        if route:  # Only non-empty routes
            color = colors[i % len(colors)]
            route_info = f"Serves {len(route)} patients"
            route_paths.append(tikz_route(route, color, i+1, route_info))
    
    # Create title comment with metrics
    title_comment = f"""% {method_name} Solution
% Served: {metrics['total_served']}, Late Pickup: {metrics['total_late_pickup']}, 
% Late Depot: {metrics['total_late_depot']}, Unserved: {metrics['total_unserved']}
% Vehicle Utilization: {metrics['vehicle_utilization']:.1f}%"""
    
    # Assemble final TikZ code
    return TIKZ_TMPL.format(
        nodes[0, 0]/10, nodes[0, 1]/10,  # Depot coordinates
        "\n".join(patient_nodes),         # Patient nodes
        title_comment + "\n" + "\n\n".join(route_paths)  # Routes with title
    )

def main():
    args = parse_args()
    
    # Problem parameters
    N = args.customers_count
    M = args.vehicles_count
    spoilage_range = args.spoilage_range
    
    # Find the most recent result directory
    results_base = "./results"
    result_dirs = []
    if os.path.exists(results_base):
        for dirname in os.listdir(results_base):
            if dirname.startswith(f"pvrp_n{N}m{M}_"):
                result_dirs.append(dirname)
    
    if not result_dirs:
        print(f"No result directories found for pvrp_n{N}m{M}")
        return
    
    # Use the most recent directory
    result_dirs.sort()
    latest_dir = result_dirs[-1]
    
    # Extract date from directory name for model path
    date = latest_dir.split('_')[-1]  # e.g., "250729-1935"
    
    model_path = f"./output/PVRPn{N}m{M}_{date}/chkpt_ep{args.epoch_count}.pyth"
    
    print(f"Generating route visualizations for PVRP N={N}, M={M}")
    print(f"Using results from: {latest_dir}")
    print(f"Model path: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Using a fresh model for demonstration")
        
    # Load learned model
    learner = AttentionLearner(
        PVRP_Dataset.CUST_FEAT_SIZE,
        PVRP_Environment.VEH_STATE_SIZE
    )
    
    if os.path.exists(model_path):
        chkpt = torch.load(model_path, map_location='cpu')
        load_old_weights(learner, chkpt["model"])
    else:
        print("Using randomly initialized model")
        
    learner.eval()
    learner.greedy = True
    
    # Generate test instance
    torch.manual_seed(12348877555)  # Fixed seed for reproducibility
    data = PVRP_Dataset.generate(1, N, M, 
                                veh_capa=args.veh_capa,
                                veh_speed=args.veh_speed,
                                spoilage_range=spoilage_range)
    
    # Get OR-Tools solution
    ort_routes = ort_solve(data)[0]
    
    # Normalize data for learned model
    nodes = data.nodes[0].clone()  # Keep original for visualization
    data.normalize()
    
    # Get learned model solution
    env = PVRP_Environment(data)
    with torch.no_grad():
        actions, _, rewards = learner(env)
    
    # Extract routes from actions
    learned_routes = [[] for _ in range(M)]
    for action in actions:
        veh_idx, cust_idx = action
        v, c = veh_idx[0].item(), cust_idx[0].item()
        if c > 0:  # Not depot
            learned_routes[v].append(c)
    
    # Remove empty routes
    learned_routes = [route for route in learned_routes if route]
    
    # Calculate costs
    ort_cost = eval_apriori_routes(env, [ort_routes], 1)[0].item()
    learned_cost = -torch.stack(rewards).sum().item()
    
    print(f"OR-Tools cost: {ort_cost:.2f}")
    print(f"Learned cost: {learned_cost:.2f}")
    print(f"Gap: {(learned_cost/ort_cost - 1)*100:+.1f}%")
    
    # Analyze route quality for both methods
    # For OR-Tools
    ort_env = PVRP_Environment(
        PVRP_Dataset(data.veh_count, data.veh_capa, data.veh_speed,
                    data.nodes.clone(), data.cust_mask)
    )
    ort_env.reset()
    for node in [n for route in [ort_routes] for n in route]:
        ort_env.step(torch.tensor([[node]], dtype=torch.long))
    if not ort_env.done:
        ort_env.step(torch.tensor([[0]], dtype=torch.long))
    
    ort_metrics, ort_late_pickup, ort_late_depot = analyze_route_quality(ort_env, [ort_routes])
    learned_metrics, learned_late_pickup, learned_late_depot = analyze_route_quality(env, learned_routes)
    
    # Find unserved nodes
    all_nodes = set(range(1, N+1))
    ort_served = set([n for route in [ort_routes] for n in route])
    learned_served = set([n for route in learned_routes for n in route])
    ort_unserved = all_nodes - ort_served
    learned_unserved = all_nodes - learned_served
    
    # Generate TikZ files
    ort_tikz = generate_route_tikz(
        nodes, [ort_routes], "OR-Tools", ort_metrics,
        ort_late_pickup, ort_late_depot, ort_unserved
    )
    
    learned_tikz = generate_route_tikz(
        nodes, learned_routes, "Learned Model", learned_metrics,
        learned_late_pickup, learned_late_depot, learned_unserved
    )
    
    # Write TikZ files with date suffix
    ort_filename = f"pvrp_ortools_n{N}m{M}_{date}.tex"
    learned_filename = f"pvrp_learned_n{N}m{M}_{date}.tex"
    
    with open(ort_filename, 'w') as f:
        f.write(ort_tikz)
    
    with open(learned_filename, 'w') as f:
        f.write(learned_tikz)
    
    print(f"\nTikZ files generated:")
    print(f"- {ort_filename}")
    print(f"- {learned_filename}")
    
    # Compile to PDF (optional)
    try:
        print("\nCompiling to PDF...")
        subprocess.run(["pdflatex", "-halt-on-error", ort_filename], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pdflatex", "-halt-on-error", learned_filename], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"PDF files generated:")
        print(f"- {ort_filename.replace('.tex', '.pdf')}")
        print(f"- {learned_filename.replace('.tex', '.pdf')}")
        
        # Open PDFs for viewing (optional)
        # subprocess.run(["xdg-open", ort_filename.replace('.tex', '.pdf')])
        # subprocess.run(["xdg-open", learned_filename.replace('.tex', '.pdf')])
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not compile LaTeX files to PDF")
        print("Make sure you have pdflatex installed")

if __name__ == "__main__":
    main()