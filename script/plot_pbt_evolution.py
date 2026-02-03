# scripts/plot_pbt_curve.py

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

def plot_pbt_with_evolution(csv_path, exploit_interval=10, output_path='pbt_evolution_marked.pdf'):
    """
    Plot PBT training curves با نشانه‌های evolution
    """
    results = load_csv(csv_path)
    
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(nrows=3, ncols=4)
    
    # Main plot
    ax = fig.add_subplot(grid[:, :3])
    
    ax.plot(results.ep, results.val, label="Observed on train data", 
            color='r', linewidth=2)
    ax.plot(results.ep, results.bl, label="Estimated by critic", 
            color='b', linewidth=2)
    if hasattr(results, 'test_mu'):
        ax.plot(results.ep, -results.test_mu, label="Observed on test data", 
                color='g', linewidth=2)
    
    # Mark exploit events
    max_epoch = int(results.ep[-1])
    for epoch in range(exploit_interval, max_epoch + 1, exploit_interval):
        ax.axvline(epoch, color='gray', alpha=0.3, linestyle=':', linewidth=1.5)
    
    # Add text annotation
    ax.text(0.02, 0.98, f'Evolution every {exploit_interval} epochs',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc="lower right", fontsize=12)
    ax.set_ylabel("Mean Cumulative Reward", fontsize=14)
    ax.set_xlabel("Training Epoch", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"PBT Training (Population=8, Exploit Interval={exploit_interval})", 
                fontsize=16, fontweight='bold')
    
    # Side plots
    metrics = [
        ('prob', 'Routes Prob.', 0),
        ('loss', 'AC Loss', 1),
        ('norm', 'Grad. Norm', 2)
    ]
    
    for metric_name, ylabel, idx in metrics:
        ax = fig.add_subplot(grid[idx, 3])
        
        ax.plot(results.ep, getattr(results, metric_name), 
                color='blue', linewidth=1.5)
        
        # Mark exploit events
        for epoch in range(exploit_interval, max_epoch + 1, exploit_interval):
            ax.axvline(epoch, color='gray', alpha=0.3, linestyle=':', linewidth=1)
        
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if idx == 2:
            ax.set_xlabel("Training Epoch", fontsize=12)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" PBT plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_pbt_curve.py <loss_gap.csv> [exploit_interval] [output_pdf]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    exploit_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    output_pdf = sys.argv[3] if len(sys.argv) > 3 else 'pbt_curve.pdf'
    
    plot_pbt_with_evolution(csv_path, exploit_interval, output_pdf)