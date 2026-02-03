#!/usr/bin/env python3
"""
Plot PBT learning curves - ROBUST VERSION
Handles CSV with # prefix in column names
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, 
                       help='Path to loss_gap.csv or directory containing it')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--show-workers', action='store_true',
                       help='Show individual worker evolution')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for saved figure')
    return parser.parse_args()


def load_csv_robust(csv_path):
    """Robustly load CSV, handling # prefix in columns"""
    
    print(f"📂 Reading CSV from: {csv_path}")
    
    # Read all lines
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Find header line (starts with #EP or EP)
    header_line = None
    data_start = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#EP') or (stripped.startswith('EP') and not stripped.startswith('#')):
            header_line = stripped
            data_start = i + 1
            break
    
    if header_line is None:
        raise ValueError("Could not find header line in CSV!")
    
    print(f"📋 Found header at line {data_start}: {header_line[:50]}...")
    
    # Parse header
    columns = header_line.split()
    columns = [col.lstrip('#') for col in columns]  # Remove # prefix
    
    print(f"📊 Columns: {columns}")
    
    # Parse data lines
    data_lines = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            data_lines.append(stripped)
    
    print(f"📊 Found {len(data_lines)} data rows")
    
    # Create DataFrame manually
    data_rows = []
    for line in data_lines:
        values = line.split()
        if len(values) == len(columns):
            data_rows.append([float(v) for v in values])
        else:
            print(f"⚠️  Skipping malformed line: {line[:50]}")
    
    df = pd.DataFrame(data_rows, columns=columns)
    
    print(f"✅ Loaded DataFrame: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   First row: {df.iloc[0].to_dict()}")
    
    return df


def plot_learning_curves(df, output_path, args):
    """Plot main learning curves"""
    
    print(f"\n📊 Creating plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], 
                          height_ratios=[1, 1, 1],
                          hspace=0.3, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[:, 0])
    ax_prob = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, 1])
    ax_norm = fig.add_subplot(gs[2, 1])
    
    # ===== MAIN PLOT =====
    
    # Check which columns we have
    has_cost_avg = 'COST_AVG' in df.columns
    has_cost_best = 'COST_BEST' in df.columns
    has_cost_worst = 'COST_WORST' in df.columns
    has_bl = 'BL' in df.columns
    has_test_mu = 'TEST_MU' in df.columns
    has_test_std = 'TEST_STD' in df.columns
    
    print(f"   Column availability:")
    print(f"     COST_AVG: {has_cost_avg}")
    print(f"     COST_BEST: {has_cost_best}")
    print(f"     COST_WORST: {has_cost_worst}")
    print(f"     BL: {has_bl}")
    print(f"     TEST_MU: {has_test_mu}")
    
    # Plot baseline (critic)
    if has_bl:
        ax_main.plot(df['EP'], -df['BL'], 'b-', 
                    label='Estimated by critic (train)', 
                    linewidth=2, alpha=0.8)
        print(f"   ✅ Plotted baseline")
    
    # Plot train costs
    if has_cost_avg:
        ax_main.plot(df['EP'], df['COST_AVG'], 'r-', 
                    label='Observed (train)', 
                    linewidth=2, alpha=0.7)
        print(f"   ✅ Plotted train average")
    
    if has_cost_best and has_cost_worst:
        # Shaded region
        ax_main.fill_between(df['EP'], df['COST_BEST'], df['COST_WORST'],
                            alpha=0.25, color='red', label='Population range')
        print(f"   ✅ Plotted population range")
    
    # Plot test
    if has_test_mu:
        ax_main.plot(df['EP'], df['TEST_MU'], 'g-', 
                    label='Observed (test)',
                    linewidth=2.5)
        print(f"   ✅ Plotted test performance")
        
        if has_test_std:
            ax_main.fill_between(df['EP'], 
                                df['TEST_MU'] - df['TEST_STD'],
                                df['TEST_MU'] + df['TEST_STD'],
                                alpha=0.2, color='green')
    
    ax_main.set_xlabel('Training epoch', fontsize=11)
    ax_main.set_ylabel('Mean cumulated reward', fontsize=11)
    ax_main.set_title('PBT Learning Curves', fontsize=13, fontweight='bold')
    ax_main.legend(loc='best', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    
    # ===== SIDE PLOTS =====
    
    # Probability
    if 'PROB' in df.columns:
        ax_prob.plot(df['EP'], df['PROB'], color='purple', linewidth=1.5)
        ax_prob.set_ylabel('Routes prob.', fontsize=10)
        ax_prob.grid(True, alpha=0.3)
        print(f"   ✅ Plotted probability")
    
    # Loss
    if 'LOSS' in df.columns:
        ax_loss.plot(df['EP'], df['LOSS'], color='orange', linewidth=1.5)
        ax_loss.set_ylabel('AC loss', fontsize=10)
        ax_loss.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax_loss.grid(True, alpha=0.3)
        print(f"   ✅ Plotted loss")
    
    # Gradient norm
    if 'NORM' in df.columns:
        ax_norm.plot(df['EP'], df['NORM'], color='brown', linewidth=1.5)
        ax_norm.set_ylabel('Grad. norm', fontsize=10)
        ax_norm.set_xlabel('Train. epoch', fontsize=10)
        ax_norm.grid(True, alpha=0.3)
        print(f"   ✅ Plotted gradient norm")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"\n✅ Saved plot to: {output_path}")
    plt.close()


def plot_worker_evolution(csv_dir, output_path, args):
    """Plot worker evolution from hyperparams logs"""
    
    import glob
    import json
    
    hyperparam_files = sorted(glob.glob(os.path.join(csv_dir, 'hyperparams_ep*.json')))
    
    if not hyperparam_files:
        print("⚠️  No hyperparams_ep*.json found. Skipping worker plots.")
        return
    
    print(f"\n📊 Creating worker evolution plots...")
    print(f"   Found {len(hyperparam_files)} hyperparam logs")
    
    # Collect data
    epochs = []
    n_workers = 8  # Default
    worker_perfs = {i: [] for i in range(n_workers)}
    worker_lrs = {i: [] for i in range(n_workers)}
    worker_tanhs = {i: [] for i in range(n_workers)}
    
    for file in hyperparam_files:
        with open(file) as f:
            data = json.load(f)
            epoch = data['epoch']
            epochs.append(epoch)
            
            for worker in data['workers']:
                wid = worker['worker_id']
                if wid not in worker_perfs:
                    worker_perfs[wid] = []
                    worker_lrs[wid] = []
                    worker_tanhs[wid] = []
                
                worker_perfs[wid].append(worker['best_performance'])
                worker_lrs[wid].append(worker['hyperparams']['actor_lr'])
                worker_tanhs[wid].append(worker['hyperparams']['tanh_xplor'])
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Performance
    ax_perf = axes[0]
    for wid in sorted(worker_perfs.keys()):
        if worker_perfs[wid]:
            ax_perf.plot(epochs, worker_perfs[wid], 
                        marker='o', label=f'Worker {wid}')
    
    ax_perf.set_xlabel('Epoch', fontsize=11)
    ax_perf.set_ylabel('Worker Performance', fontsize=11)
    ax_perf.set_title('Individual Worker Performance (PBT)', 
                     fontsize=13, fontweight='bold')
    ax_perf.legend(ncol=2, fontsize=9)
    ax_perf.grid(True, alpha=0.3)
    
    # Plot 2: Learning rates
    ax_lr = axes[1]
    for wid in sorted(worker_lrs.keys()):
        if worker_lrs[wid]:
            ax_lr.plot(epochs, worker_lrs[wid], 
                      marker='x', alpha=0.7, label=f'Worker {wid}')
    
    ax_lr.set_xlabel('Epoch', fontsize=11)
    ax_lr.set_ylabel('Actor Learning Rate', fontsize=11)
    ax_lr.set_title('Learning Rate Evolution', fontsize=12, fontweight='bold')
    ax_lr.grid(True, alpha=0.3)
    
    # Plot 3: Tanh exploration
    ax_tanh = axes[2]
    for wid in sorted(worker_tanhs.keys()):
        if worker_tanhs[wid]:
            ax_tanh.plot(epochs, worker_tanhs[wid], 
                        marker='s', alpha=0.7, label=f'Worker {wid}')
    
    ax_tanh.set_xlabel('Epoch', fontsize=11)
    ax_tanh.set_ylabel('Tanh Exploration', fontsize=11)
    ax_tanh.set_title('Exploration Parameter Evolution', 
                     fontsize=12, fontweight='bold')
    ax_tanh.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    worker_output = output_path.replace('.pdf', '_workers.pdf')
    plt.savefig(worker_output, dpi=args.dpi, bbox_inches='tight')
    print(f"✅ Saved worker plots to: {worker_output}")
    plt.close()


def main():
    args = parse_args()
    
    # Handle directory vs file input
    if os.path.isdir(args.csv_path):
        csv_path = os.path.join(args.csv_path, 'loss_gap.csv')
        csv_dir = args.csv_path
    else:
        csv_path = args.csv_path
        csv_dir = os.path.dirname(csv_path)
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        sys.exit(1)
    
    # Load data
    try:
        df = load_csv_robust(csv_path)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        output_path = csv_path.replace('.csv', '.pdf')
    else:
        output_path = args.output
    
    # Plot main curves
    try:
        plot_learning_curves(df, output_path, args)
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Plot worker evolution
    if args.show_workers:
        try:
            plot_worker_evolution(csv_dir, output_path, args)
        except Exception as e:
            print(f"⚠️  Error creating worker plots: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ All done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()