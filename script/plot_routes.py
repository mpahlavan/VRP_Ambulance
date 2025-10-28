#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PVRP Analyzer with Corrected Cost Calculation
"""

import os, time, numpy as np, torch
from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import *
from marpdan.dep import matplotlib as mpl, pyplot as plt
from torch.utils.data import DataLoader

mpl.rcParams["axes.titlesize"] = 20
SEED = 12348877555
ROLLOUTS = 800
BATCH_SIZE = 100

plt.style.use('seaborn-v0_8')
mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.titlesize': 15,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})


def compute_late_sets(env, batch_idx=0):
    """
    Extract late sets from environment state for a specific batch instance
    
    Args:
        env: PVRP_Environment
        batch_idx: which instance in the batch to analyze
    
    Returns:
        late_nodes: set of nodes with late pickup
        late_depot: set of nodes with late hospital arrival
    """
    if not hasattr(env, "vehicles"):
        return set(), set()
    
    # Late pickup nodes
    late_nodes = set()
    if getattr(env, "late_nodes", None) is not None:
        late_mask = env.late_nodes[batch_idx]  # Get specific batch instance
        late_indices = torch.nonzero(late_mask, as_tuple=False).flatten().tolist()
        late_nodes = set(late_indices)
        late_nodes.discard(0)  # Remove depot
    
    # Late depot delivery
    late_depot = set()
    arrival_in_depot = env.vehicles[batch_idx, :, 3]  # Arrival times for batch_idx
    spoilage_times = env.nodes[batch_idx, :, 3]       # Survival times for batch_idx
    
    for v in range(env.veh_count):
        # Get nodes served by vehicle v in batch_idx
        nodes_for_vehicle = torch.nonzero(
            env.vehicle_routes[batch_idx] == v, as_tuple=False
        ).flatten()
        
        if nodes_for_vehicle.numel() == 0:
            continue
            
        t_arrival_depot = arrival_in_depot[v].item()
        
        for n in nodes_for_vehicle.tolist():
            if n > 0 and t_arrival_depot > spoilage_times[n]:  # Exclude depot
                late_depot.add(n)
    
    return late_nodes, late_depot






class PVRPAnalyzer:
    def __init__(self, args):
        self.problem_type = args.problem_type
        self.n_customers = args.customers_count
        self.n_vehicles = args.vehicles_count
        self.veh_capa = args.veh_capa
        self.veh_speed = args.veh_speed
        self.min_cust_count = args.min_cust_count
        self.cust_loc_range = args.loc_range
        self.horizon = args.horizon
        self.spoilage_range = args.spoilage_range
        date = "250727-1714"
        self.MODEL_PATH = f"./output/PVRPn{args.customers_count}m{args.vehicles_count}_{date}/chkpt_ep{args.epoch_count}.pyth"
        self.learner = self._load_model()
        
        self.colors = {
            'ortools': '#2E86AB',
            'learned_greedy': '#A23B72',
            'learned_sampling': '#F18F01',
            'improvement': '#2E8B57',
            'degradation': '#DC143C',
            'hospital': '#FF6B6B',
            'patient_urgent': '#FF4757',
            'patient_normal': '#3742FA',
            'late_pickup': '#FF3838',
            'late_depot': '#FF6348',
            'unserved': '#8B0000',
            'infeasible': '#696969'
        }
        
    def _load_model(self):
        try:
            chkpt = torch.load(self.MODEL_PATH, map_location="cpu")
            learner = AttentionLearner(
                cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
                veh_state_size=PVRP_Environment.VEH_STATE_SIZE
            )
            learner.load_state_dict(chkpt["model"])
            learner.eval()
            return learner
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")
    
    def generate_data(self):
        torch.manual_seed(SEED)
        data = PVRP_Dataset.generate(
            BATCH_SIZE, 
            self.n_customers, 
            self.n_vehicles, 
            self.veh_capa,
            self.veh_speed,
            self.min_cust_count,
            self.cust_loc_range,
            self.horizon,
            self.spoilage_range
        )
        ref_routes = ort_solve(data)
        data.normalize()
        return data, ref_routes

    def calculate_reference_costs(self, data, ref_routes):
        """محاسبه هزینه‌های OR-Tools با environment (شامل penalties)"""
        ref_costs = []
        
        for idx, routes in enumerate(ref_routes):
            env = PVRP_Environment(
                data, 
                nodes=data.nodes[idx:idx+1],  # ← FIX: instance صحیح
                cust_mask=data.cust_mask[idx:idx+1] if data.cust_mask is not None else None
            )
            env.reset()
            rewards = []
            
            for route in routes:
                for node in route:
                    rewards.append(env.step(torch.tensor([[node]], dtype=torch.long)))
            
            if not env.done:
                rewards.append(env.step(torch.tensor([[0]], dtype=torch.long)))
            
            ref_costs.append(-torch.stack(rewards).sum())
        
        return torch.stack(ref_costs)

    def evaluate_learned_model_batch(self, data):
        """Evaluate learned model using batch processing"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learner.to(device)
        data.nodes = data.nodes.to(device)
        if data.cust_mask is not None:
            data.cust_mask = data.cust_mask.to(device)
        
        # GREEDY evaluation
        print(" Evaluating Greedy Ambulance Routing...")
        self.learner.greedy = True
        
        env = PVRP_Environment(data)
        with torch.no_grad():
            _, logps, rewards = self.learner(env)
        
        greedy_costs = torch.stack(rewards).sum(0).mul(-1).squeeze(1).cpu()
        greedy_probs = torch.stack(logps).sum(0).squeeze(1).exp().cpu()
        
        print(f"greedy {greedy_costs.mean():.3f} +- {greedy_costs.std():.3f} w.p. {greedy_probs.mean():.3g}")

        # SAMPLING evaluation  
        print(" Evaluating Sampling Ambulance Routing...")
        self.learner.greedy = False
        
        roll_costs = []
        roll_logps = []
        for _ in range(ROLLOUTS):
            env = PVRP_Environment(data)
            with torch.no_grad():
                _, logp, rewards = self.learner(env)
            roll_costs.append(torch.stack(rewards).sum(0).mul(-1).squeeze(1))
            roll_logps.append(torch.stack(logp).sum(0).squeeze(1))
        
        best_cost, best_idx = torch.stack(roll_costs).min(0, keepdim=True)
        sampling_costs = best_cost.squeeze(0).cpu()
        sampling_probs = torch.stack(roll_logps).gather(0, best_idx).squeeze(0).exp().cpu()
        
        print(f"sample {sampling_costs.mean():.3f} +- {sampling_costs.std():.3f} w.p. {sampling_probs.mean():.3g}")
        
        return greedy_costs, sampling_costs

    @staticmethod
    def plot_pvrp_instance(ax, nodes, routes, title, 
                          late_nodes=None, late_depot_nodes=None, 
                          unserved_nodes=None, infeasible_nodes=None, colors=None):
        """Enhanced plot with ambulance theme"""
        late_nodes = late_nodes or set()
        late_depot_nodes = late_depot_nodes or set()
        unserved_nodes = unserved_nodes or set()
        infeasible_nodes = infeasible_nodes or set()
        colors = colors or {}
        
        # Hospital
        ax.plot(nodes[0,0].item(), nodes[0,1].item(), 
                marker='H', markersize=15, color=colors.get('hospital', '#FF6B6B'), 
                markeredgecolor='black', markeredgewidth=2,
                label=' Hospital')
        
        # Patients
        if len(nodes) > 1:
             scatter = ax.scatter(nodes[1:,0], nodes[1:,1], 
                             c=nodes[1:,3], 
                             cmap='RdYlGn',
                             edgecolors="k", linewidths=0.4,
                             s=180, 
                             label='patient (color=survival time)')
             plt.colorbar(scatter, ax=ax, shrink=0.8)
        
        # Routes
        route_styles = ['-', '--', '-.', ':']
        route_colors = plt.cm.Set1(np.linspace(0, 1, max(len(routes), 1)))
        
        for route_idx, (route, color) in enumerate(zip(routes, route_colors)):
            if not route:
                continue
                
            full_route = [0] + route + [0]
            points = nodes[full_route]
            
            style = route_styles[route_idx % len(route_styles)]
            ax.plot(points[:,0], points[:,1], style, 
                   color=color, alpha=0.8, linewidth=3,
                   label=f' Route {route_idx+1}')
            
            for i in range(len(points)-1):
                ax.annotate('', xy=(points[i+1, 0], points[i+1, 1]), 
                           xytext=(points[i, 0], points[i, 1]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))
        
        late_depot_only = late_depot_nodes - late_nodes
        
        # Infeasible
        if infeasible_nodes:
            for n in infeasible_nodes:
                ax.scatter(nodes[n, 0], nodes[n, 1],
                        marker='^', s=250, facecolors='none',
                        edgecolors='red', linewidths=2.5,
                        label=" Infeasible" if n == min(infeasible_nodes) else "",
                        zorder=6)
        
        # Late pickup
        if late_nodes:
            for n in late_nodes:
                ax.scatter(nodes[n, 0], nodes[n, 1],
                        marker='o', s=220, facecolors='none',
                        edgecolors='red', linewidths=3.0,
                        label="Late pickup" if n == min(late_nodes) else "",
                        zorder=7)
        
        # Late hospital
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
                        label=" Unserved" if n == min(unserved_nodes) else "",
                        zorder=8)
        
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect("equal")
        
        legend = ax.legend(loc="upper left", fontsize=9, frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

    def plot_cost_analysis(self, ref_costs, greedy_costs, sampling_costs, outdir):
        """Create comprehensive cost analysis plots"""
        # Calculate gaps
        greedy_gaps = greedy_costs / ref_costs - 1
        sampling_gaps = sampling_costs / ref_costs - 1
        
        # Calculate statistics
        ref_mean = ref_costs.mean().item()
        greedy_mean = greedy_costs.mean().item()
        sampling_mean = sampling_costs.mean().item()
        
        greedy_gap_mean = greedy_gaps.mean().item()
        sampling_gap_mean = sampling_gaps.mean().item()
        
        greedy_improvement_rate = (greedy_gaps < 0).sum().item() / len(greedy_gaps) * 100
        sampling_improvement_rate = (sampling_gaps < 0).sum().item() / len(sampling_gaps) * 100
        
        # Main dashboard
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig1.suptitle(' Ambulance Routing Performance Dashboard ', fontsize=18, y=0.96, fontweight='bold')
        
        # Box plot comparison
        bp = ax1.boxplot([ref_costs.numpy(), greedy_costs.numpy(), sampling_costs.numpy()],
                         labels=[' OR-Tools', ' RL Greedy', ' RL Sampling'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['ortools'])
        bp['boxes'][1].set_facecolor(self.colors['learned_greedy'])
        bp['boxes'][2].set_facecolor(self.colors['learned_sampling'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        ax1.set_title(' Cost Distribution Comparison', fontsize=14, pad=15, fontweight='bold')
        ax1.set_ylabel('Emergency Response Cost', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        ax1.text(1, ref_mean + ref_costs.std().item() * 0.3, f'μ={ref_mean:.1f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['ortools'], alpha=0.8))
        ax1.text(2, greedy_mean + greedy_costs.std().item() * 0.3, f'μ={greedy_mean:.1f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['learned_greedy'], alpha=0.8))
        ax1.text(3, sampling_mean + sampling_costs.std().item() * 0.3, f'μ={sampling_mean:.1f}', 
                 ha='center', va='bottom', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['learned_sampling'], alpha=0.8))
        
        # Gap histogram
        ax2.hist([greedy_gaps.numpy() * 100, sampling_gaps.numpy() * 100], 
                bins=20, alpha=0.7, 
                color=[self.colors['learned_greedy'], self.colors['learned_sampling']],
                label=[' Greedy', ' Sampling'],
                edgecolor='black', linewidth=0.5)
        ax2.axvline(greedy_gap_mean * 100, color=self.colors['learned_greedy'], linestyle='--', linewidth=2,
                    label=f'Greedy Mean: {greedy_gap_mean:+.1%}')
        ax2.axvline(sampling_gap_mean * 100, color=self.colors['learned_sampling'], linestyle='--', linewidth=2,
                    label=f'Sampling Mean: {sampling_gap_mean:+.1%}')
        ax2.axvline(0, color='blue', linestyle='-', alpha=0.8, linewidth=2,
                    label=' OR-Tools baseline')
        
        ax2.set_title(' Performance Gap Distribution', fontsize=14, pad=15, fontweight='bold')
        ax2.set_xlabel('Performance Gap (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Scatter plot
        scatter_colors_greedy = [self.colors['improvement'] if g < 0 else self.colors['degradation'] 
                                for g in greedy_gaps.numpy()]
        scatter_colors_sampling = [self.colors['improvement'] if g < 0 else self.colors['degradation'] 
                                  for g in sampling_gaps.numpy()]
        
        ax3.scatter(ref_costs.numpy(), greedy_costs.numpy(), 
                   c=scatter_colors_greedy, alpha=0.7, s=45, 
                   edgecolors='black', linewidth=0.3, label=' Greedy')
        ax3.scatter(ref_costs.numpy(), sampling_costs.numpy(), 
                   c=scatter_colors_sampling, alpha=0.7, s=45, marker='^',
                   edgecolors='black', linewidth=0.3, label=' Sampling')
        
        min_cost = min(ref_costs.min().item(), greedy_costs.min().item(), sampling_costs.min().item())
        max_cost = max(ref_costs.max().item(), greedy_costs.max().item(), sampling_costs.max().item())
        ax3.plot([min_cost, max_cost], [min_cost, max_cost], 'k--', alpha=0.8,
                linewidth=2.5, label='📏 Equal Performance')
        
        ax3.set_xlabel(' OR-Tools Cost', fontsize=12)
        ax3.set_ylabel(' RL Agent Cost', fontsize=12)
        ax3.set_title(' Direct Cost Comparison', fontsize=14, pad=15, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left', fontsize=9)
        
        # Performance summary
        ax4.text(0.5, 0.9, ' Performance Summary', transform=ax4.transAxes, 
                ha='center', va='top', fontsize=16, fontweight='bold')
        
        summary_text = f"""
 OR-Tools Avg Cost: {ref_mean:.1f}

 Greedy Ambulance:
  • Avg Cost: {greedy_mean:.1f}
  • Avg Gap: {greedy_gap_mean:+.1%}
  • Better Cases: {greedy_improvement_rate:.1f}%

 Sampling Ambulance:
  • Avg Cost: {sampling_mean:.1f}
  • Avg Gap: {sampling_gap_mean:+.1%}
  • Better Cases: {sampling_improvement_rate:.1f}%

 Best Strategy: {"Sampling" if sampling_gap_mean < greedy_gap_mean else "Greedy"}
        """
        
        ax4.text(0.1, 0.8, summary_text, transform=ax4.transAxes,
                fontsize=12, va='top', ha='left', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, 
                           wspace=0.25, hspace=0.35)
        
        cost_plot_path = f"{outdir}/ambulance_performance_dashboard.pdf"
        fig1.savefig(cost_plot_path, bbox_inches="tight", dpi=300, facecolor='white')
        plt.close(fig1)
        
        # Simple comparison plot
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig2.suptitle(f' Ambulance Routing Performance Summary ', fontsize=14, fontweight='bold')
        
        # Bar chart
        means = [ref_mean, greedy_mean, sampling_mean]
        stds = [ref_costs.std().item(), greedy_costs.std().item(), sampling_costs.std().item()]
        labels = [' OR-Tools', ' RL Greedy', ' RL Sampling']
        bar_colors = [self.colors['ortools'], self.colors['learned_greedy'], self.colors['learned_sampling']]
        
        bars = ax1.bar(labels, means, yerr=stds, capsize=5,
                       color=bar_colors, alpha=0.7, edgecolor='black')
        ax1.set_title(' Average Response Cost', fontweight='bold')
        ax1.set_ylabel('Emergency Response Cost', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        for bar, mean, std in zip(bars, means, stds):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.1,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Gap comparison
        gap_data = [greedy_gaps.numpy() * 100, sampling_gaps.numpy() * 100]
        gap_labels = [' Greedy', ' Sampling']
        gap_colors = [self.colors['learned_greedy'], self.colors['learned_sampling']]
        
        bp2 = ax2.boxplot(gap_data, labels=gap_labels, patch_artist=True)
        for box, color in zip(bp2['boxes'], gap_colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
        ax2.set_title(' Performance Gap Distribution', fontweight='bold')
        ax2.set_ylabel('Performance Gap (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        simple_plot_path = f"{outdir}/ambulance_comparison_simple.pdf"
        fig2.savefig(simple_plot_path, bbox_inches="tight", dpi=300)
        plt.close(fig2)
        
        print(f"\n Enhanced ambulance analysis plots created:")
        print(f"   • {cost_plot_path}")
        print(f"   • {simple_plot_path}")

    def analyze_and_visualize(self):
        """تحلیل و بصری‌سازی کامل"""
        
        data, ref_routes = self.generate_data()
        ref_costs = self.calculate_reference_costs(data, ref_routes)
        
        greedy_costs, sampling_costs = self.evaluate_learned_model_batch(data)
        
        greedy_gaps = greedy_costs / ref_costs - 1
        sampling_gaps = sampling_costs / ref_costs - 1
        
        print("\n Emergency Response Cost Analysis:")
        print(f"Average OR-Tools Cost: {ref_costs.mean().item():.2f}")
        print(f"Average Greedy Cost: {greedy_costs.mean().item():.2f} (Gap: {greedy_gaps.mean():.2%})")
        print(f"Average Sampling Cost: {sampling_costs.mean().item():.2f} (Gap: {sampling_gaps.mean():.2%})")
        
        output_dir = f"results/ambulance_n{self.n_customers}m{self.n_vehicles}_{time.strftime('%y%m%d-%H%M')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot cost analysis
        self.plot_cost_analysis(ref_costs, greedy_costs, sampling_costs, output_dir)
        
        print(" Creating route visualizations...")
        
        greedy_gaps_sorted, sub_idx = greedy_gaps.sort()
        
        display_indices = torch.cat((
            sub_idx[:4],
            sub_idx[BATCH_SIZE//2-2:BATCH_SIZE//2+2],
            sub_idx[-4:]
        ))
        
        # Run learned model for all instances to get routes
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learner.greedy = True
        
        full_env = PVRP_Environment(data)
        with torch.no_grad():
            learned_actions, _, _ = self.learner(full_env)
        
        for i, idx in enumerate(display_indices):
            cust = data.nodes[idx]
            ref_route = ref_routes[idx]
            ref_cost = ref_costs[idx].item()
            model_cost = greedy_costs[idx].item()
            gap = model_cost / ref_cost - 1
            
            # Extract learned routes for this instance
            learned_routes = [[] for _ in range(data.veh_count)]
            for action in learned_actions:
                veh_idx, node_idx = action
                v = veh_idx[idx].item()
                n = node_idx[idx].item()
                if n > 0:
                    learned_routes[v].append(n)
            learned_routes = [route for route in learned_routes if route]
            
            late_lea, late_dep_lea = compute_late_sets(full_env, batch_idx=idx.item())
            
            # Compute served/unserved sets
            ref_served = set([node for route in ref_route for node in route])
            learned_served = set([node for route in learned_routes for node in route])
            all_nodes = set(range(1, len(cust)))
            
            # Compute infeasible nodes
            depot_pos = cust[0:1, :2]
            node_pos = cust[:, :2]
            to_node_dist = torch.norm(node_pos - depot_pos, dim=-1)
            round_trip_time = 2 * (to_node_dist / data.veh_speed)
            infeasible_mask = round_trip_time > cust[:, 3]
            infeasible_nodes = set(torch.nonzero(infeasible_mask).flatten().tolist())
            infeasible_nodes.discard(0)  # Ensure depot is not in infeasible
            
            ref_unserved = (all_nodes - ref_served) - infeasible_nodes
            learned_unserved = (all_nodes - learned_served) - infeasible_nodes
            
            # Dummy late sets for OR-Tools visualization (simplified)
            surv = cust[:, 3]
            mean_s = surv[1:].mean()
            late_ref = set(torch.nonzero(surv < mean_s * 0.7).flatten().tolist())
            late_dep_ref = set(torch.nonzero(surv < mean_s * 0.5).flatten().tolist())
            late_ref.discard(0)  # Ensure depot is not in late sets
            late_dep_ref.discard(0)
            
            fig, (ref_ax, ax) = plt.subplots(1, 2, figsize=(24, 10))
            
            ref_title = (f" OR-Tools (Cost: {ref_cost:.1f})\n"
                         f" Served: {len(ref_served)} | Unserved: {len(ref_unserved)} | "
                         f"Infeasible: {len(infeasible_nodes)}")
            
            learned_title = (f" RL Routing (Cost: {model_cost:.1f}, Gap: {gap:.0%})\n"
                            f" Served: {len(learned_served)} | Late Pickup: {len(late_lea)} | "
                            f"Late Hospital: {len(late_dep_lea)} | Unserved: {len(learned_unserved)} | "
                            f"Infeasible: {len(infeasible_nodes)}")
            
            self.plot_pvrp_instance(ref_ax, cust, ref_route, ref_title,
                                   late_nodes=late_ref, late_depot_nodes=late_dep_ref,
                                   unserved_nodes=ref_unserved, infeasible_nodes=infeasible_nodes,
                                   colors=self.colors)
            self.plot_pvrp_instance(ax, cust, learned_routes, learned_title,
                                   late_nodes=late_lea, late_depot_nodes=late_dep_lea,
                                   unserved_nodes=learned_unserved, infeasible_nodes=infeasible_nodes,
                                   colors=self.colors)
            
            fig.tight_layout()
            file_path = f"{output_dir}/routes_{i:02}_{100*gap:.0f}.pdf"
            fig.savefig(file_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        print(f"\n Analysis complete! Results in: {output_dir}/")


def main(args):
    analyzer = PVRPAnalyzer(args)
    analyzer.analyze_and_visualize()

if __name__ == "__main__":
    main(parse_args())