#!/usr/bin/env python3
# pbt_trainer.py - Complete PBT Trainer with ALL FIXES

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from itertools import chain
import numpy as np
import os
import json
import csv
import copy

from marpdan.dep import tqdm


class PBTWorker:
    """Single worker in PBT population"""
    
    def __init__(self, worker_id, learner, baseline, args, device):
        self.worker_id = worker_id
        self.device = device
        self.args = args
        
        # Deep copy models
        self.learner = copy.deepcopy(learner)
        self.baseline = copy.deepcopy(baseline) if baseline is not None else None
        
        # Move to device
        self.learner.to(device)
        if self.baseline is not None:
            self.baseline.to(device)
        
        # Initialize hyperparameters with variation
        self.hyperparams = self._sample_initial_hyperparams()
        
        # Create optimizer
        self._create_optimizer()
        
        # Performance tracking
        self.best_performance = float('inf')
        self.performance_history = []
        
    def _sample_initial_hyperparams(self):
        """Sample initial hyperparameters with variation"""
        import numpy as np
        
        base_actor_lr = self.args.learning_rate
        base_critic_lr = self.args.critic_rate
        
        # Sample with ±50% variation
        actor_lr = base_actor_lr * np.random.uniform(0.5, 2.0)
        critic_lr = base_critic_lr * np.random.uniform(0.5, 2.0)
        
        # Sample exploration parameter
        tanh_xplor = self.args.tanh_xplor + np.random.uniform(-2, 2)
        tanh_xplor = np.clip(tanh_xplor, 3, 15)
        
        return {
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'tanh_xplor': tanh_xplor
        }
    
    def _create_optimizer(self):
        """Create optimizer with current hyperparameters"""
        if self.baseline is not None:
            self.optimizer = Adam([
                {"params": self.learner.parameters(), 
                 "lr": self.hyperparams['actor_lr']},
                {"params": self.baseline.parameters(), 
                 "lr": self.hyperparams['critic_lr']}
            ])
        else:
            self.optimizer = Adam(
                self.learner.parameters(), 
                self.hyperparams['actor_lr']
            )
    
    def update_hyperparams(self, new_hyperparams):
        """Update hyperparameters and recreate optimizer"""
        self.hyperparams = new_hyperparams.copy()
        self._create_optimizer()
    
    def get_performance(self):
        """Get worker performance (lower is better for cost)"""
        return self.best_performance
    
    def update_performance(self, performance):
        """Update performance tracking"""
        self.performance_history.append(performance)
        if performance < self.best_performance:
            self.best_performance = performance


class PBTTrainer:
    """Population-Based Training for PVRP"""
    
    def __init__(self, args, dataset_class, env_class, learner_class, 
                 baseline_class, device, n_workers):
        self.args = args
        self.dataset_class = dataset_class
        self.env_class = env_class
        self.learner_class = learner_class
        self.baseline_class = baseline_class
        self.device = device
        self.n_workers = n_workers
        
        # Reference costs (for gap computation)
        self.ref_costs = None
        self.test_env = None
        
        # Initialize population
        print(f"\n{'='*60}")
        print(f"Initializing Population with {n_workers} workers")
        print(f"{'='*60}\n")
        
        self._initialize_population()
        
    def _initialize_population(self):
        """Initialize population of workers"""
        # Create base models
        base_learner = self.learner_class(
            self.dataset_class.CUST_FEAT_SIZE,
            self.env_class.VEH_STATE_SIZE,
            self.args.model_size,
            self.args.layer_count,
            self.args.head_count,
            self.args.ff_size,
            self.args.tanh_xplor
        )
        
        base_baseline = self.baseline_class(
            base_learner,
            self.args.customers_count,
            self.args.critic_use_qval,
            self.args.loss_use_cumul
        )
        
        # Create workers
        self.workers = []
        for i in range(self.n_workers):
            worker = PBTWorker(i, base_learner, base_baseline, 
                             self.args, self.device)
            self.workers.append(worker)
            
            print(f"Worker {i}: actor_lr={worker.hyperparams['actor_lr']:.6f}, "
                  f"critic_lr={worker.hyperparams['critic_lr']:.6f}, "
                  f"tanh={worker.hyperparams['tanh_xplor']:.1f}")
        
        print()
    
    def set_reference_costs(self, ref_costs):
        """Set OR-Tools reference costs for gap computation"""
        self.ref_costs = ref_costs
        print(f"Reference costs set: mean={ref_costs.mean():.2f} ± {ref_costs.std():.2f}")
    
    def train_epoch_worker(self, worker, train_data, env_params):
        """Train a single worker for one epoch"""
        
        worker.learner.train()
        # Note: baseline doesn't need explicit train() - it's handled internally
        
        loader = DataLoader(train_data, self.args.batch_size, shuffle=True)
        
        epoch_losses = []
        epoch_vals = []
        epoch_probs = []
        epoch_bls = []
        epoch_norms = []
        
        # Apply current tanh exploration
        worker.learner.C = worker.hyperparams['tanh_xplor']
        
        for minibatch in loader:
            if train_data.cust_mask is None:
                custs, mask = minibatch.to(self.device), None
            else:
                custs, mask = minibatch[0].to(self.device), minibatch[1].to(self.device)
            
            # Create environment
            dyna = self.env_class(train_data, custs, mask, *env_params)
            
            # Forward pass
            actions, logps, rewards, bl_vals = worker.baseline(dyna)
            
            # Compute loss
            from marpdan.layers import reinforce_loss
            loss = reinforce_loss(logps, rewards, bl_vals)
            
            # Backward pass
            worker.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = clip_grad_norm_(
                chain.from_iterable(grp["params"] for grp in worker.optimizer.param_groups),
                self.args.max_grad_norm
            )
            
            worker.optimizer.step()
            
            # Metrics
            prob = torch.stack(logps).sum(0).exp().mean()
            val = rewards.mean() if isinstance(rewards, torch.Tensor) else torch.stack(rewards).sum(0).mean()
            bl = bl_vals[0].mean()
            
            epoch_losses.append(loss.item())
            epoch_vals.append(val.item())
            epoch_probs.append(prob.item())
            epoch_bls.append(bl.item())
            epoch_norms.append(grad_norm)
        
        # Return averages
        return (
            np.mean(epoch_losses),
            np.mean(epoch_vals),
            np.mean(epoch_probs),
            np.mean(epoch_bls),
            np.mean(epoch_norms)
        )
    
    def evaluate_worker(self, worker, test_env):
        """Evaluate worker on test environment"""
        worker.learner.eval()
        
        with torch.no_grad():
            _, _, rewards = worker.learner(test_env)
            costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
        
        test_cost = costs.mean().item()
        return test_cost
    
    def exploit_and_explore(self, performances):
        """PBT exploit and explore operation"""
        
        # Sort workers by performance (lower cost is better)
        sorted_indices = np.argsort(performances)
        
        # Bottom 25% exploit top 25%
        n_bottom = max(1, len(self.workers) // 4)
        n_top = max(1, len(self.workers) // 4)
        
        bottom_indices = sorted_indices[-n_bottom:]  # Worst workers
        top_indices = sorted_indices[:n_top]  # Best workers
        
        print(f"\n  🔄 Exploit & Explore:")
        print(f"    Top workers: {top_indices.tolist()} (perf: {[performances[i] for i in top_indices]})")
        print(f"    Bottom workers: {bottom_indices.tolist()} (perf: {[performances[i] for i in bottom_indices]})")
        
        for bottom_idx in bottom_indices:
            # Choose random top worker to copy from
            top_idx = np.random.choice(top_indices)
            
            print(f"    Worker {bottom_idx} exploits Worker {top_idx}")
            
            # Copy model weights
            self.workers[bottom_idx].learner.load_state_dict(
                self.workers[top_idx].learner.state_dict()
            )
            if self.workers[bottom_idx].baseline is not None:
                self.workers[bottom_idx].baseline.load_state_dict(
                    self.workers[top_idx].baseline.state_dict()
                )
            
            # Copy and perturb hyperparameters
            new_hyperparams = {}
            for key, value in self.workers[top_idx].hyperparams.items():
                if key.endswith('_lr'):
                    # Perturb learning rate by ×0.8 or ×1.2
                    factor = np.random.choice([0.8, 1.2])
                    new_val = value * factor
                    new_val = np.clip(new_val, 1e-6, 1e-2)
                    new_hyperparams[key] = new_val
                elif key == 'tanh_xplor':
                    # Perturb exploration by ±2
                    new_val = value + np.random.uniform(-2, 2)
                    new_val = np.clip(new_val, 3, 15)
                    new_hyperparams[key] = new_val
                else:
                    new_hyperparams[key] = value
            
            self.workers[bottom_idx].update_hyperparams(new_hyperparams)
    
    def train(self, n_epochs, exploit_interval=10, mini_epochs=10):
        """Main PBT training loop"""
        
        print(f"\n{'='*60}")
        print(f"Starting PBT Training")
        print(f"Total Epochs: {n_epochs}")
        print(f"Exploit Interval: {exploit_interval}")
        print(f"Mini Epochs per Step: {mini_epochs}")
        print(f"{'='*60}\n")
        
        # Setup CSV logging
        csv_path = os.path.join(self.args.output_dir, 'loss_gap.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter='\t')
        
        # Updated header with best/worst
        csv_writer.writerow([
            '#EP', '#LOSS', '#COST_AVG', '#COST_BEST', '#COST_WORST',
            '#PROB', '#BL', '#NORM', 
            '#TEST_MU', '#TEST_STD', '#TEST_GAP'
        ])
        
        print(f"Logging to: {csv_path}\n")
        
        # Environment parameters (fixed, not evolved)
        env_params = [
            self.args.spoilage_penalty,
            self.args.unserved_penalty,
            self.args.dist_penalty_coef,
            0,  # pickup_bonus_coef (removed)
            self.args.additional_late_penalty,
            self.args.capacity_usage_coef,
            self.args.idle_penalty_coef
        ]
        
        best_worker = None
        best_performance = float('inf')
        
        for ep in range(n_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {ep+1}/{n_epochs}")
            print(f"{'='*60}")
            
            # ============================================================
            # CRITICAL: Generate FRESH training data every epoch
            # ============================================================
            print(f"🔄 Generating fresh training data...")
            train_data = self.dataset_class.generate(
                self.args.iter_count * self.args.batch_size,
                self.args.customers_count,
                self.args.vehicles_count,
                self.args.veh_capa,
                self.args.veh_speed,
                self.args.min_cust_count,
                self.args.loc_range,
                self.args.horizon,
                self.args.spoilage_range
            )
            train_data.normalize()
            print(f"✅ Generated {train_data.batch_size} fresh instances\n")
            
            # Train all workers
            losses, vals, probs, bls, norms = [], [], [], [], []
            
            for w_idx, worker in enumerate(self.workers):
                loss, val, prob, bl, norm = self.train_epoch_worker(
                    worker, train_data, env_params
                )
                
                losses.append(loss)
                vals.append(val)
                probs.append(prob)
                bls.append(bl)
                norms.append(norm)
                
                # Convert to cost (positive)
                cost = -val
                
                print(f"Worker {w_idx}: Loss={loss:.4f}, Cost={cost:.2f}, "
                      f"Prob={prob:.3f}, GradNorm={norm:.4f}")
            
            # Population statistics
            costs = [-v for v in vals]  # Convert rewards to costs
            avg_cost = np.mean(costs)
            best_cost = np.min(costs)
            worst_cost = np.max(costs)
            
            print(f"\nPopulation Average: Loss={np.mean(losses):.4f}, "
                  f"Cost={avg_cost:.2f}, Best={best_cost:.2f}, "
                  f"Worst={worst_cost:.2f}, BL={np.mean(bls):.2f}, "
                  f"Norm={np.mean(norms):.2f}")
            
            # Evaluate on test set
            print(f"\nEvaluating on test set...")
            test_performances = []
            for w_idx, worker in enumerate(self.workers):
                test_cost = self.evaluate_worker(worker, self.test_env)
                test_performances.append(test_cost)
                worker.update_performance(test_cost)
                
                if test_cost < best_performance:
                    best_performance = test_cost
                    best_worker = worker
                    print(f"  Worker {w_idx}: Test Performance={test_cost:.2f}")
                    print(f"    🌟 New global best!")
                else:
                    print(f"  Worker {w_idx}: Test Performance={test_cost:.2f}")
            
            # Compute test statistics
            test_mu = np.mean(test_performances)
            test_std = np.std(test_performances)
            
            if self.ref_costs is not None:
                test_gap = (test_mu / self.ref_costs.mean().item() - 1.0)
            else:
                test_gap = 0.0
            
            print(f"\nTest Average: Cost={test_mu:.2f} ± {test_std:.2f}, "
                  f"Gap={test_gap:.2%}")
            
            # Log to CSV
            csv_writer.writerow([
                ep+1,
                np.mean(losses),
                avg_cost,
                best_cost,
                worst_cost,
                np.mean(probs),
                np.mean(bls),
                np.mean(norms),
                test_mu,
                test_std,
                test_gap
            ])
            csv_file.flush()
            
            # Exploit & Explore
            if (ep + 1) % exploit_interval == 0 and ep < n_epochs - 1:
                self.exploit_and_explore(test_performances)
            
            # Save checkpoint
            if (ep + 1) % self.args.checkpoint_period == 0:
                self.save_checkpoint(ep + 1, best_worker)
        
        csv_file.close()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Worker: {best_worker.worker_id}")
        print(f"Best Performance: {best_performance:.2f}")
        print(f"{'='*60}")
        
        return best_worker
    
    def save_checkpoint(self, epoch, best_worker):
        """Save checkpoint at given epoch"""
        
        # Save full population
        pop_dir = os.path.join(self.args.output_dir, f'population_epoch_{epoch}')
        os.makedirs(pop_dir, exist_ok=True)
        
        for worker in self.workers:
            torch.save({
                'learner': worker.learner.state_dict(),
                'baseline': worker.baseline.state_dict() if worker.baseline else None,
                'hyperparams': worker.hyperparams,
                'performance': worker.best_performance
            }, os.path.join(pop_dir, f'worker_{worker.worker_id}.pt'))
        
        # Save best worker in compatible format
        chkpt_path = os.path.join(self.args.output_dir, f'chkpt_ep{epoch}.pyth')
        torch.save({
            'learner': best_worker.learner.state_dict(),
            'baseline': best_worker.baseline.state_dict() if best_worker.baseline else None,
            'epoch': epoch
        }, chkpt_path)
        
        # Save hyperparameters log
        hyperparams_log = {
            'epoch': epoch,
            'workers': [
                {
                    'worker_id': w.worker_id,
                    'hyperparams': w.hyperparams,
                    'best_performance': w.best_performance
                }
                for w in self.workers
            ]
        }
        
        hyperparams_path = os.path.join(self.args.output_dir, f'hyperparams_ep{epoch}.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams_log, f, indent=2)
        
        print(f"  Population saved to {pop_dir}")
        print(f"  Compatible checkpoint saved to {chkpt_path}")
        print(f"  Hyperparameters log saved to {hyperparams_path}")
    
    def save_final_results(self, best_worker):
        """Save final training results"""
        
        # Save best worker
        best_path = os.path.join(self.args.output_dir, 'best_worker.pt')
        torch.save({
            'learner': best_worker.learner.state_dict(),
            'baseline': best_worker.baseline.state_dict() if best_worker.baseline else None,
            'hyperparams': best_worker.hyperparams,
            'performance': best_worker.best_performance,
            'worker_id': best_worker.worker_id
        }, best_path)
        
        # Save compatible format
        compat_path = os.path.join(self.args.output_dir, f'chkpt_ep{self.args.epoch_count}.pyth')
        torch.save({
            'learner': best_worker.learner.state_dict(),
            'baseline': best_worker.baseline.state_dict() if best_worker.baseline else None,
            'epoch': self.args.epoch_count
        }, compat_path)
        
        # Save summary
        summary = {
            'best_worker_id': best_worker.worker_id,
            'best_performance': best_worker.best_performance,
            'best_hyperparams': best_worker.hyperparams,
            'all_workers_final': [
                {
                    'worker_id': w.worker_id,
                    'final_performance': w.best_performance,
                    'hyperparams': w.hyperparams
                }
                for w in self.workers
            ]
        }
        
        summary_path = os.path.join(self.args.output_dir, 'pbt_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Final Results Saved:")
        print(f"  PBT format: {best_path}")
        print(f"  Compatible format: {compat_path}")
        print(f"  Summary: {summary_path}")
        print(f"  CSV Log: {os.path.join(self.args.output_dir, 'loss_gap.csv')}")
        print(f"{'='*60}")