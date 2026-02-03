# neuroevolution/evolution_ops.py

import numpy as np
import torch

class EvolutionOperations:
    """
    عملیات Exploit و Explore برای PBT
    """
    
    @staticmethod
    def exploit(workers, exploit_ratio=0.25):
        """
        Exploit: بدترین workers از بهترین‌ها copy می‌کنند

        Args:
            workers: لیست PBTWorker
            exploit_ratio: نسبت workers برای exploit (0.25 = bottom 25%)

        Returns:
            indices بهترین و بدترین workers
        """
        # Sort by performance (LOWER cost is better, so NO reverse)
        sorted_indices = sorted(
            range(len(workers)),
            key=lambda i: workers[i].get_performance(),
            reverse=False  # Changed from True to False for cost minimization
        )
        
        n_exploit = max(1, int(len(workers) * exploit_ratio))
        
        top_indices = sorted_indices[:n_exploit]
        bottom_indices = sorted_indices[-n_exploit:]
        
        # Bottom workers copy از top workers
        for bottom_idx, top_idx in zip(bottom_indices, top_indices):
            print(f"  Worker {bottom_idx} (perf={workers[bottom_idx].get_performance():.2f}) "
                  f"copies from Worker {top_idx} (perf={workers[top_idx].get_performance():.2f})")
            
            workers[bottom_idx].copy_from(workers[top_idx])
        
        return top_indices, bottom_indices
    
    @staticmethod
    def explore(workers, bottom_indices, perturb_factors=[0.5, 2.0]):
        """
        Explore: mutate کردن hyperparameters

        Args:
            workers: لیست PBTWorker
            bottom_indices: indices workers که باید mutate شوند
            perturb_factors: factors برای perturbation
        """
        for idx in bottom_indices:
            worker = workers[idx]
            old_hyperparams = worker.hyperparams.copy()

            # Perturb هر hyperparameter
            new_hyperparams = {}
            for key, value in old_hyperparams.items():
                if key.endswith('_lr'):  # Learning rates
                    # Multiply by random factor (increased range for more exploration)
                    factor = np.random.choice(perturb_factors)
                    new_val = value * factor
                    # Clip به range معقول
                    new_val = np.clip(new_val, 1e-6, 1e-2)
                    new_hyperparams[key] = new_val

                elif key == 'tanh_xplor':
                    # Perturb با additive noise (increased range)
                    new_val = value + np.random.uniform(-3, 3)
                    new_val = np.clip(new_val, 3, 15)
                    new_hyperparams[key] = new_val

                elif key.endswith('_penalty') or key.endswith('_coef'):
                    # Multiply by factor
                    factor = np.random.choice(perturb_factors)
                    new_val = value * factor
                    new_val = np.clip(new_val, 0.01, 10.0)
                    new_hyperparams[key] = new_val

                else:
                    # Keep unchanged
                    new_hyperparams[key] = value

            # Update worker
            worker.update_hyperparams(new_hyperparams)

            print(f"  Worker {idx} mutated:")
            for key in new_hyperparams:
                if new_hyperparams[key] != old_hyperparams[key]:
                    print(f"    {key}: {old_hyperparams[key]:.6f} → {new_hyperparams[key]:.6f}")