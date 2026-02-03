#!/usr/bin/env python3
"""
Comprehensive PBT Visualization for Paper
Generates all plots needed for PBT section of thesis/paper

Usage:
    python plot_pbt_analysis.py <pbt_output_dir>
    
Outputs:
    1. pbt_hyperparams_evolution.pdf - How hyperparameters evolved
    2. pbt_worker_performance.pdf - Individual worker performance
    3. pbt_population_diversity.pdf - Population diversity over time
    4. pbt_summary_table.tex - LaTeX table with final results
"""

import json
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Style for paper
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def load_hyperparams_logs(pbt_dir):
    """Load all hyperparameter logs from PBT run"""
    pattern = os.path.join(pbt_dir, 'hyperparams_ep*.json')
    files = glob.glob(pattern)

    if not files:
        print(f"⚠️  No hyperparams_ep*.json found in {pbt_dir}")
        return None

    # Sort numerically by epoch number (not alphabetically)
    def get_epoch(f):
        basename = os.path.basename(f)
        # Extract number from "hyperparams_ep123.json"
        return int(basename.replace('hyperparams_ep', '').replace('.json', ''))

    files = sorted(files, key=get_epoch)

    data = []
    for f in files:
        with open(f) as fp:
            data.append(json.load(fp))

    print(f"✅ Loaded {len(data)} hyperparameter logs")
    return data


def load_pbt_summary(pbt_dir):
    """Load PBT summary file"""
    summary_path = os.path.join(pbt_dir, 'pbt_summary.json')
    
    if not os.path.exists(summary_path):
        print(f"⚠️  pbt_summary.json not found")
        return None
    
    with open(summary_path) as f:
        return json.load(f)


def plot_hyperparams_evolution(data, output_path):
    """Plot how hyperparameters evolved during PBT"""
    
    if data is None:
        return
    
    epochs = [d['epoch'] for d in data]
    n_workers = len(data[0]['workers'])
    
    # Collect hyperparameters for each worker
    actor_lrs = {i: [] for i in range(n_workers)}
    critic_lrs = {i: [] for i in range(n_workers)}
    tanhs = {i: [] for i in range(n_workers)}
    
    for d in data:
        for w in d['workers']:
            wid = w.get('worker_id', w.get('id'))  # Handle both key names
            actor_lrs[wid].append(w['hyperparams']['actor_lr'])
            critic_lrs[wid].append(w['hyperparams']['critic_lr'])
            tanhs[wid].append(w['hyperparams']['tanh_xplor'])
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_workers))
    
    # Plot 1: Actor LR
    ax = axes[0]
    for wid in range(n_workers):
        ax.plot(epochs, actor_lrs[wid], 'o-', color=colors[wid], 
                label=f'Worker {wid}', alpha=0.7, markersize=4)
    ax.set_ylabel('Actor LR ($\\eta_\\pi$)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, loc='upper right', fontsize=8)
    ax.set_title('Hyperparameter Evolution During PBT', fontweight='bold')
    
    # Plot 2: Critic LR
    ax = axes[1]
    for wid in range(n_workers):
        ax.plot(epochs, critic_lrs[wid], 's-', color=colors[wid], 
                alpha=0.7, markersize=4)
    ax.set_ylabel('Critic LR ($\\eta_V$)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Tanh exploration
    ax = axes[2]
    for wid in range(n_workers):
        ax.plot(epochs, tanhs[wid], '^-', color=colors[wid], 
                alpha=0.7, markersize=4)
    ax.set_ylabel('Exploration (C)')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_worker_performance(data, output_path):
    """Plot individual worker performance over time"""
    
    if data is None:
        return
    
    epochs = [d['epoch'] for d in data]
    n_workers = len(data[0]['workers'])
    
    # Collect performance
    performances = {i: [] for i in range(n_workers)}
    
    for d in data:
        for w in d['workers']:
            wid = w.get('worker_id', w.get('id'))  # Handle both key names
            performances[wid].append(w.get('best_performance', w.get('performance')))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_workers))
    
    for wid in range(n_workers):
        ax.plot(epochs, performances[wid], 'o-', color=colors[wid],
                label=f'Worker {wid}', linewidth=2, markersize=5, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best Performance (Cost)')
    ax.set_title('Individual Worker Performance During PBT', fontweight='bold')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower cost is better

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_population_diversity(data, output_path):
    """Plot population diversity metrics"""
    
    if data is None:
        return
    
    epochs = [d['epoch'] for d in data]
    
    # Calculate diversity metrics
    perf_std = []
    actor_lr_std = []
    critic_lr_std = []
    tanh_std = []
    
    for d in data:
        perfs = [w.get('best_performance', w.get('performance')) for w in d['workers']]
        actor_lrs = [w['hyperparams']['actor_lr'] for w in d['workers']]
        critic_lrs = [w['hyperparams']['critic_lr'] for w in d['workers']]
        tanhs = [w['hyperparams']['tanh_xplor'] for w in d['workers']]
        
        perf_std.append(np.std(perfs))
        actor_lr_std.append(np.std(actor_lrs) / np.mean(actor_lrs))  # CV
        critic_lr_std.append(np.std(critic_lrs) / np.mean(critic_lrs))
        tanh_std.append(np.std(tanhs))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    
    # Performance diversity
    ax = axes[0, 0]
    ax.plot(epochs, perf_std, 'b-o', linewidth=2, markersize=5)
    ax.set_ylabel('Std Dev')
    ax.set_title('Performance Diversity')
    ax.grid(True, alpha=0.3)
    
    # Actor LR diversity (CV)
    ax = axes[0, 1]
    ax.plot(epochs, actor_lr_std, 'r-s', linewidth=2, markersize=5)
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Actor LR Diversity')
    ax.grid(True, alpha=0.3)
    
    # Critic LR diversity (CV)
    ax = axes[1, 0]
    ax.plot(epochs, critic_lr_std, 'g-^', linewidth=2, markersize=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Critic LR Diversity')
    ax.grid(True, alpha=0.3)
    
    # Tanh diversity
    ax = axes[1, 1]
    ax.plot(epochs, tanh_std, 'm-d', linewidth=2, markersize=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Std Dev')
    ax.set_title('Exploration Parameter Diversity')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Population Diversity During PBT', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Saved: {output_path}")
    plt.close()


def generate_latex_table(summary, output_path):
    """Generate LaTeX table with PBT results"""
    
    if summary is None:
        return
    
    best_hp = summary['best_hyperparams']
    best_perf = summary['best_performance']
    best_id = summary['best_worker_id']
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Optimized Hyperparameters from Population-Based Training}
\label{tab:pbt_results}
\begin{tabular}{lc}
\toprule
\textbf{Parameter} & \textbf{Optimized Value} \\
\midrule
Actor learning rate ($\eta_\pi$) & %.2e \\
Critic learning rate ($\eta_V$) & %.2e \\
Exploration temperature ($C$) & %.2f \\
\midrule
Best worker ID & %d \\
Best validation cost & %.4f \\
\bottomrule
\end{tabular}
\end{table}
""" % (best_hp['actor_lr'], best_hp['critic_lr'], best_hp['tanh_xplor'],
       best_id, best_perf)
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved: {output_path}")
    
    # Also print to console
    print("\n" + "="*50)
    print("LaTeX Table Preview:")
    print("="*50)
    print(f"Actor LR:   {best_hp['actor_lr']:.2e}")
    print(f"Critic LR:  {best_hp['critic_lr']:.2e}")
    print(f"Tanh:       {best_hp['tanh_xplor']:.2f}")
    print(f"Best Cost:  {best_perf:.4f}")
    print("="*50 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_pbt_analysis.py <pbt_output_dir>")
        print("\nExample:")
        print("  python plot_pbt_analysis.py output/PVRPn10m2_PBT8_251113-0928")
        sys.exit(1)
    
    pbt_dir = sys.argv[1]
    
    if not os.path.exists(pbt_dir):
        print(f"❌ Error: {pbt_dir} not found!")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"PBT Analysis for Paper")
    print(f"{'='*60}")
    print(f"Directory: {pbt_dir}\n")
    
    # Load data
    hp_data = load_hyperparams_logs(pbt_dir)
    summary = load_pbt_summary(pbt_dir)
    
    # Generate plots
    print("\n📊 Generating plots...\n")
    
    plot_hyperparams_evolution(
        hp_data, 
        os.path.join(pbt_dir, 'pbt_hyperparams_evolution.pdf')
    )
    
    plot_worker_performance(
        hp_data,
        os.path.join(pbt_dir, 'pbt_worker_performance.pdf')
    )
    
    plot_population_diversity(
        hp_data,
        os.path.join(pbt_dir, 'pbt_population_diversity.pdf')
    )
    
    generate_latex_table(
        summary,
        os.path.join(pbt_dir, 'pbt_results_table.tex')
    )
    
    print(f"\n{'='*60}")
    print(f"✅ All outputs saved to: {pbt_dir}")
    print(f"{'='*60}\n")
    
    print("Files generated:")
    print("  📈 pbt_hyperparams_evolution.pdf")
    print("  📈 pbt_worker_performance.pdf")
    print("  📈 pbt_population_diversity.pdf")
    print("  📝 pbt_results_table.tex")


if __name__ == "__main__":
    main()