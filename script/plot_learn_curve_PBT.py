#!/usr/bin/env python3
#plot_learn_curve_PBT.py

"""
Plot learning curves for Population-Based Training (PBT)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
from argparse import ArgumentParser
import json
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("result_path",
                       help="Path to PBT output directory or loss_gap.csv file")
    parser.add_argument("--output-path", "-o", default=None,
                       help="Path to the pdf file the plot will be exported to")
    parser.add_argument("--font-size", default=20,
                       help="Reference size of fonts for all title/label on the figure")
    parser.add_argument("--show-workers", action="store_true",
                       help="Show individual worker performance (if hyperparams logs available)")
    return parser.parse_args()


def load_pbt_results(result_path):
    """Load results from PBT directory or CSV file"""
    
    # Check if it's a directory or file
    if os.path.isdir(result_path):
        csv_path = os.path.join(result_path, "loss_gap.csv")
        hyperparam_dir = result_path
    else:
        csv_path = result_path
        hyperparam_dir = os.path.dirname(result_path)
    
    # Load CSV
    with open(csv_path) as f:
        header = next(f)
        Result = namedtuple("Result", 
                          [col.strip(" \n#").lower() for col in header.split() if col])
        results = []
        for row in f:
            results.append(Result(*(float(val) for val in row.split() if val)))
    
    results = Result(*(np.array(res) for res in zip(*results)))

    # Handle different CSV formats: old format has 'val', new PBT format has 'cost_avg'
    # If val doesn't exist, create it from -cost_avg
    if not hasattr(results, 'val') and hasattr(results, 'cost_avg'):
        # Create a new namedtuple with val field added
        ResultWithVal = namedtuple('ResultWithVal', results._fields + ('val',))
        results = ResultWithVal(*results, -results.cost_avg)

    # Load hyperparameter logs if available
    hyperparam_logs = []
    hyperparam_files = sorted([f for f in os.listdir(hyperparam_dir) 
                              if f.startswith("hyperparams_ep") and f.endswith(".json")])
    
    for hf in hyperparam_files:
        with open(os.path.join(hyperparam_dir, hf)) as f:
            hyperparam_logs.append(json.load(f))
    
    return results, hyperparam_logs


def plot_standard_curves(results, args):
    """Plot standard learning curves (like original plot_learn_curve.py)"""
    
    mpl.rc('font', size=args.font_size)
    
    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(nrows=3, ncols=4)
    
    # Main plot: rewards
    ax = fig.add_subplot(grid[:, :3])
    ax.plot(results.ep, results.val, label="Observed (train)", color='r', linewidth=2)
    ax.plot(results.ep, results.bl, label="Estimated by critic (train)", color='b', linewidth=2)
    
    if not np.isnan(results.test_mu).all():
        ax.plot(results.ep, -results.test_mu, label="Observed (test)", color='g', linewidth=2)
        ax.fill_between(results.ep, 
                        -results.test_mu - results.test_std,
                        -results.test_mu + results.test_std, 
                        color='g', alpha=0.3)
    
    ax.legend(loc="lower right")
    ax.set_ylabel("Mean cumulated reward")
    ax.set_xlabel("Training epoch")
    ax.set_title("PBT Learning Curves")
    ax.grid(True, alpha=0.3)
    
    # Probability
    ax = fig.add_subplot(grid[0, 3])
    ax.plot(results.ep, results.prob, color='purple', linewidth=2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Routes prob.")
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = fig.add_subplot(grid[1, 3])
    ax.plot(results.ep, np.abs(results.loss), color='orange', linewidth=2)


    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("AC loss")
    ax.grid(True, alpha=0.3)
    
    # Gradient norm
    ax = fig.add_subplot(grid[2, 3])
    ax.plot(results.ep, results.norm, color='brown', linewidth=2)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Grad. norm")
    ax.set_xlabel("Train. epoch")
    ax.grid(True, alpha=0.3)
    
    fig.set_size_inches(16, 9)
    
    return fig


def plot_pbt_specific(results, hyperparam_logs, args):
    """Plot PBT-specific visualizations"""
    
    mpl.rc('font', size=args.font_size)
    
    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(nrows=2, ncols=2)
    
    # Plot 1: Worker performance over time
    ax = fig.add_subplot(grid[0, :])
    
    if hyperparam_logs:
        # Extract worker performances
        epochs = [log['epoch'] for log in hyperparam_logs]
        n_workers = len(hyperparam_logs[0]['workers'])
        
        for worker_id in range(n_workers):
            perfs = []
            for log in hyperparam_logs:
                worker_data = [w for w in log['workers'] if w['id'] == worker_id]
                if worker_data:
                    perfs.append(worker_data[0]['performance'])
                else:
                    perfs.append(np.nan)
            
            ax.plot(epochs, perfs, label=f"Worker {worker_id}", alpha=0.7, marker='o')
        
        ax.legend(loc="best", ncol=2)
        ax.set_ylabel("Worker Performance")
        ax.set_xlabel("Epoch")
        ax.set_title("Individual Worker Performance (PBT)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No hyperparameter logs available", 
               ha='center', va='center', transform=ax.transAxes)
    
    # Plot 2: Hyperparameter evolution (actor_lr)
    ax = fig.add_subplot(grid[1, 0])
    
    if hyperparam_logs:
        epochs = [log['epoch'] for log in hyperparam_logs]
        
        for worker_id in range(n_workers):
            lrs = []
            for log in hyperparam_logs:
                worker_data = [w for w in log['workers'] if w['id'] == worker_id]
                if worker_data:
                    lrs.append(worker_data[0]['hyperparams']['actor_lr'])
                else:
                    lrs.append(np.nan)
            
            ax.plot(epochs, lrs, label=f"Worker {worker_id}", alpha=0.7, marker='x')
        
        ax.set_ylabel("Actor Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_title("Learning Rate Evolution")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Hyperparameter evolution (tanh_xplor)
    ax = fig.add_subplot(grid[1, 1])
    
    if hyperparam_logs:
        for worker_id in range(n_workers):
            tanhs = []
            for log in hyperparam_logs:
                worker_data = [w for w in log['workers'] if w['id'] == worker_id]
                if worker_data:
                    tanhs.append(worker_data[0]['hyperparams']['tanh_xplor'])
                else:
                    tanhs.append(np.nan)
            
            ax.plot(epochs, tanhs, label=f"Worker {worker_id}", alpha=0.7, marker='s')
        
        ax.set_ylabel("Tanh Exploration")
        ax.set_xlabel("Epoch")
        ax.set_title("Exploration Parameter Evolution")
        ax.grid(True, alpha=0.3)
    
    fig.set_size_inches(16, 10)
    
    return fig


def main(args):
    # Load results
    print(f"Loading results from: {args.result_path}")
    results, hyperparam_logs = load_pbt_results(args.result_path)
    print(f"  Found {len(results.ep)} epochs")
    if hyperparam_logs:
        print(f"  Found {len(hyperparam_logs)} hyperparameter logs")
    
    # Plot standard curves
    fig1 = plot_standard_curves(results, args)
    
    # Save
    if args.output_path is None:
        if os.path.isdir(args.result_path):
            args.output_path = os.path.join(args.result_path, "learning_curves.pdf")
        else:
            args.output_path = args.result_path.replace(".csv", ".pdf")
    
    fig1.savefig(args.output_path, dpi=300, bbox_inches='tight')
    print(f"Saved standard curves to: {args.output_path}")
    
    # Plot PBT-specific if requested and data available
    if args.show_workers and hyperparam_logs:
        fig2 = plot_pbt_specific(results, hyperparam_logs, args)
        
        pbt_path = args.output_path.replace(".pdf", "_pbt_workers.pdf")
        fig2.savefig(pbt_path, dpi=300, bbox_inches='tight')
        print(f"Saved PBT worker curves to: {pbt_path}")
    
    plt.show()


if __name__ == "__main__":
    main(parse_args())