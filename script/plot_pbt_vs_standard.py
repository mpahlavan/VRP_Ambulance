# scripts/compare_pbt_standard.py

import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import sys

def load_csv(csv_path):
    """Load loss_gap.csv"""
    with open(csv_path) as f:
        header = next(f)
        Result = namedtuple("Result", 
                           [col.strip(" \n#").lower() for col in header.split() if col])
        results = []
        for row in f:
            try:
                results.append(Result(*(float(val) for val in row.split() if val)))
            except:
                continue
    return Result(*(np.array(res) for res in zip(*results)))

def plot_comparison(pbt_csv, standard_csv, output_path='comparison.pdf'):
    """
    مقایسه PBT vs Standard REINFORCE
    
    Args:
        pbt_csv: path to PBT loss_gap.csv
        standard_csv: path to Standard REINFORCE loss_gap.csv
        output_path: output PDF path
    """
    # Load data
    pbt = load_csv(pbt_csv)
    standard = load_csv(standard_csv)
    
    # Create figure
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(nrows=3, ncols=4)
    
    # Main plot: Reward
    ax = fig.add_subplot(grid[:, :3])
    
    # PBT
    ax.plot(pbt.ep, pbt.val, label="PBT (train)", color='blue', linewidth=2)
    ax.plot(pbt.ep, pbt.bl, label="PBT (critic)", color='cyan', 
            linewidth=2, linestyle='--', alpha=0.7)
    if hasattr(pbt, 'test_mu'):
        ax.plot(pbt.ep, -pbt.test_mu, label="PBT (test)", color='green', linewidth=2)
    
    # Standard REINFORCE
    ax.plot(standard.ep, standard.val, label="Standard (train)", 
            color='red', linewidth=2, alpha=0.8)
    ax.plot(standard.ep, standard.bl, label="Standard (critic)", 
            color='orange', linewidth=2, linestyle='--', alpha=0.7)
    if hasattr(standard, 'test_mu'):
        ax.plot(standard.ep, -standard.test_mu, label="Standard (test)", 
                color='purple', linewidth=2, alpha=0.8)
    
    ax.legend(loc="lower right", fontsize=12)
    ax.set_ylabel("Mean Cumulative Reward", fontsize=14)
    ax.set_xlabel("Training Epoch", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_title("PBT vs Standard REINFORCE", fontsize=16, fontweight='bold')
    
    # Side plots
    metrics = [
        ('prob', 'Routes Prob.', 0),
        ('loss', 'AC Loss', 1),
        ('norm', 'Grad. Norm', 2)
    ]
    
    for metric_name, ylabel, idx in metrics:
        ax = fig.add_subplot(grid[idx, 3])
        
        ax.plot(pbt.ep, getattr(pbt, metric_name), 
                label='PBT', color='blue', linewidth=1.5)
        ax.plot(standard.ep, getattr(standard, metric_name), 
                label='Standard', color='red', linewidth=1.5, linestyle='--')
        
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=10)
        if idx == 2:
            ax.set_xlabel("Training Epoch", fontsize=12)
    
    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Comparison plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_pbt_standard.py <pbt_csv> <standard_csv> [output_pdf]")
        sys.exit(1)
    
    pbt_csv = sys.argv[1]
    standard_csv = sys.argv[2]
    output_pdf = sys.argv[3] if len(sys.argv) > 3 else 'pbt_vs_standard.pdf'
    
    plot_comparison(pbt_csv, standard_csv, output_pdf)