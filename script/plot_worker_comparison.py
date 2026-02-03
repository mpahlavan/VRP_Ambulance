# scripts/plot_worker_comparison.py

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_worker_comparison(summary_file, output='worker_comparison.pdf'):
    """
     performance of workers
    """
    with open(summary_file) as f:
        summary = json.load(f)
    
    workers = summary['all_workers_final']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Final Performance
    ax = axes[0]
    worker_ids = [w['id'] for w in workers]
    final_perfs = [w['final_performance'] for w in workers]
    best_perfs = [w['best_performance'] for w in workers]
    
    x = np.arange(len(worker_ids))
    width = 0.35
    
    ax.bar(x - width/2, final_perfs, width, label='Final', alpha=0.8)
    ax.bar(x + width/2, best_perfs, width, label='Best Ever', alpha=0.8)
    
    ax.set_xlabel('Worker ID')
    ax.set_ylabel('Performance (Reward)')
    ax.set_title('Worker Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(worker_ids)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best worker
    best_id = summary['best_worker']['id']
    ax.axvline(best_id, color='gold', linestyle='--', linewidth=2, 
              label=f'Best (W{best_id})')
    
    # Plot 2: Learning Rate Distribution
    ax = axes[1]
    actor_lrs = [w['hyperparams']['actor_lr'] for w in workers]
    critic_lrs = [w['hyperparams']['critic_lr'] for w in workers]
    
    ax.scatter(actor_lrs, critic_lrs, s=100, alpha=0.6, c=final_perfs, 
              cmap='RdYlGn_r')
    
    # Annotate with worker IDs
    for i, wid in enumerate(worker_ids):
        ax.annotate(f'W{wid}', (actor_lrs[i], critic_lrs[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Actor LR')
    ax.set_ylabel('Critic LR')
    ax.set_title('Learning Rate Space')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # # Plot 3: Penalty Coefficients
    # ax = axes[2]
    # spoilage = [w['hyperparams']['spoilage_penalty'] for w in workers]
    # pickup = [w['hyperparams']['pickup_bonus_coef'] for w in workers]
    
    # sc = ax.scatter(spoilage, pickup, s=100, alpha=0.6, c=final_perfs,
    #                cmap='RdYlGn_r')
    
    # for i, wid in enumerate(worker_ids):
    #     ax.annotate(f'W{wid}', (spoilage[i], pickup[i]),
    #                xytext=(5, 5), textcoords='offset points')
    
    # ax.set_xlabel('Spoilage Penalty')
    # ax.set_ylabel('Pickup Bonus Coef')
    # ax.set_title('Reward Coefficient Space')
    # ax.grid(True, alpha=0.3)
    
    # plt.colorbar(sc, ax=ax, label='Performance')
    
    # plt.tight_layout()
    # plt.savefig(output, dpi=300, bbox_inches='tight')
    # print(f" Saved to {output}")
    # plt.show()

#