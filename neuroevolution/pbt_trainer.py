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
from marpdan.layers import reinforce_loss


def env_kwargs_from_args(args):
    """Keyword arguments for PVRP_Environment, matched by name."""
    return dict(
        spoilage_penalty=args.spoilage_penalty,
        unserved_penalty=args.unserved_penalty,
        dist_penalty_coef=args.dist_penalty_coef,
        additional_late_penalty=args.additional_late_penalty,
        idle_penalty_coef=args.idle_penalty_coef,
    )


class PBTWorker:
    """Single worker in PBT population"""

    def __init__(self, worker_id, learner, baseline, args, device):
        self.worker_id = worker_id
        self.device = device
        self.args = args

        self.learner = copy.deepcopy(learner)
        self.baseline = copy.deepcopy(baseline) if baseline is not None else None

        self.learner.to(device)
        if self.baseline is not None:
            # FIX 1: the critic wrapper must use this worker's actor
            self.baseline.learner = self.learner
            self.baseline.to(device)

        self.hyperparams = self._sample_initial_hyperparams()
        self.apply_exploration()
        self._create_optimizer()

        self.best_performance = float('inf')
        self.last_performance = float('inf')
        self.performance_history = []

    def _sample_initial_hyperparams(self):
        """Initial values: learning rates x U(0.5, 2.0), C + U(-2, 2) clipped to [3, 15]."""
        base_actor_lr = self.args.learning_rate
        base_critic_lr = self.args.critic_rate
        actor_lr = base_actor_lr * np.random.uniform(0.5, 2.0)
        critic_lr = base_critic_lr * np.random.uniform(0.5, 2.0)
        tanh_xplor = float(np.clip(self.args.tanh_xplor + np.random.uniform(-2, 2), 3, 15))
        return {'actor_lr': actor_lr, 'critic_lr': critic_lr, 'tanh_xplor': tanh_xplor}

    def apply_exploration(self):
        # FIX 2: AttentionLearner reads self.tanh_xplor
        self.learner.tanh_xplor = self.hyperparams['tanh_xplor']

    def _create_optimizer(self):
        if self.baseline is not None:
            self.optimizer = Adam([
                {"params": self.learner.parameters(), "lr": self.hyperparams['actor_lr']},
                {"params": self.baseline.parameters(), "lr": self.hyperparams['critic_lr']},
            ])
        else:
            self.optimizer = Adam(self.learner.parameters(), self.hyperparams['actor_lr'])

    def update_hyperparams(self, new_hyperparams):
        self.hyperparams = new_hyperparams.copy()
        self.apply_exploration()
        self._create_optimizer()

    def get_performance(self):
        return self.best_performance

    def update_performance(self, performance):
        self.last_performance = performance
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
        self.env_kwargs = env_kwargs_from_args(args)

        self.ref_costs = None
        self.test_env = None
        self.global_best = None      # snapshot of the best (worker, epoch) seen

        print(f"\n{'='*60}")
        print(f"Initializing Population with {n_workers} workers")
        print(f"{'='*60}\n")
        print(f"Environment kwargs: {self.env_kwargs}\n")
        self._initialize_population()

    def _initialize_population(self):
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

        self.workers = []
        for i in range(self.n_workers):
            worker = PBTWorker(i, base_learner, base_baseline, self.args, self.device)
            # sanity check for FIX 1
            assert worker.baseline is None or worker.baseline.learner is worker.learner
            self.workers.append(worker)
            print(f"Worker {i}: actor_lr={worker.hyperparams['actor_lr']:.6f}, "
                  f"critic_lr={worker.hyperparams['critic_lr']:.6f}, "
                  f"tanh={worker.hyperparams['tanh_xplor']:.2f}")
        print()

    def set_reference_costs(self, ref_costs):
        self.ref_costs = ref_costs

    def make_env(self, data, nodes=None, mask=None):
        # FIX 3: keyword construction
        return self.env_class(data, nodes, mask, **self.env_kwargs)

    def train_epoch_worker(self, worker, train_data):
        worker.learner.train()
        worker.apply_exploration()
        loader = DataLoader(train_data, self.args.batch_size, shuffle=True)

        epoch_losses, epoch_vals, epoch_probs, epoch_bls, epoch_norms = [], [], [], [], []

        for minibatch in loader:
            if train_data.cust_mask is None:
                custs, mask = minibatch.to(self.device), None
            else:
                custs, mask = minibatch[0].to(self.device), minibatch[1].to(self.device)

            dyna = self.make_env(train_data, custs, mask)
            actions, logps, rewards, bl_vals = worker.baseline(dyna)
            loss = reinforce_loss(logps, rewards, bl_vals)

            worker.optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(
                chain.from_iterable(grp["params"] for grp in worker.optimizer.param_groups),
                self.args.max_grad_norm
            )
            worker.optimizer.step()

            prob = torch.stack(logps).sum(0).exp().mean()
            val = rewards.mean() if isinstance(rewards, torch.Tensor) else torch.stack(rewards).sum(0).mean()
            bl = bl_vals[0].mean()

            epoch_losses.append(loss.item())
            epoch_vals.append(val.item())
            epoch_probs.append(prob.item())
            epoch_bls.append(bl.item())
            epoch_norms.append(float(grad_norm))

        return (np.mean(epoch_losses), np.mean(epoch_vals), np.mean(epoch_probs),
                np.mean(epoch_bls), np.mean(epoch_norms))

    def evaluate_worker(self, worker, test_env):
        """Greedy evaluation on the validation set (deterministic ranking)."""
        worker.learner.eval()
        worker.apply_exploration()
        old_greedy = getattr(worker.learner, 'greedy', False)
        worker.learner.greedy = True
        with torch.no_grad():
            _, _, rewards = worker.learner(test_env)
            costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
        worker.learner.greedy = old_greedy
        return costs.mean().item()

    def exploit_and_explore(self, performances):
        sorted_indices = np.argsort(performances)
        n_bottom = max(1, len(self.workers) // 4)
        n_top = max(1, len(self.workers) // 4)
        bottom_indices = sorted_indices[-n_bottom:]
        top_indices = sorted_indices[:n_top]

        print(f"\n  Exploit & Explore:")
        print(f"    Top workers: {top_indices.tolist()} (perf: {[round(performances[i], 3) for i in top_indices]})")
        print(f"    Bottom workers: {bottom_indices.tolist()} (perf: {[round(performances[i], 3) for i in bottom_indices]})")

        for bottom_idx in bottom_indices:
            top_idx = np.random.choice(top_indices)
            print(f"    Worker {bottom_idx} exploits Worker {top_idx}")
            src, dst = self.workers[top_idx], self.workers[bottom_idx]

            dst.learner.load_state_dict(src.learner.state_dict())
            if dst.baseline is not None:
                dst.baseline.load_state_dict(src.baseline.state_dict())

            new_hyperparams = {}
            for key, value in src.hyperparams.items():
                if key.endswith('_lr'):
                    factor = np.random.choice([0.8, 1.2])
                    new_hyperparams[key] = float(np.clip(value * factor, 1e-6, 1e-2))
                elif key == 'tanh_xplor':
                    new_hyperparams[key] = float(np.clip(value + np.random.uniform(-2, 2), 3, 15))
                else:
                    new_hyperparams[key] = value
            dst.update_hyperparams(new_hyperparams)
            # the copied weights belong to the source; reset the record
            dst.best_performance = float('inf')

    def train(self, n_epochs, exploit_interval=10, mini_epochs=10):
        print(f"\n{'='*60}")
        print(f"Starting PBT Training")
        print(f"Total Epochs: {n_epochs}")
        print(f"Exploit Interval: {exploit_interval}")
        print(f"{'='*60}\n")

        csv_path = os.path.join(self.args.output_dir, 'loss_gap.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow([
            '#EP', '#LOSS', '#COST_AVG', '#COST_BEST', '#COST_WORST',
            '#PROB', '#BL', '#NORM',
            '#TEST_MU', '#TEST_STD', '#TEST_BEST', '#TEST_GAP'
        ])

        hp_path = os.path.join(self.args.output_dir, 'hyperparams_history.csv')
        hp_file = open(hp_path, 'w', newline='')
        hp_writer = csv.writer(hp_file, delimiter='\t')
        hp_writer.writerow(['#EP', '#WORKER', '#ACTOR_LR', '#CRITIC_LR', '#TANH', '#VAL_COST'])

        for ep in range(n_epochs):
            print(f"\n{'='*60}\nEpoch {ep+1}/{n_epochs}\n{'='*60}")

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

            losses, vals, probs, bls, norms = [], [], [], [], []
            for w_idx, worker in enumerate(self.workers):
                loss, val, prob, bl, norm = self.train_epoch_worker(worker, train_data)
                losses.append(loss); vals.append(val); probs.append(prob)
                bls.append(bl); norms.append(norm)
                print(f"Worker {w_idx}: Loss={loss:.4f}, Cost={-val:.3f}, "
                      f"Prob={prob:.3f}, GradNorm={norm:.4f}")

            costs = [-v for v in vals]
            print(f"\nPopulation: Cost avg={np.mean(costs):.3f} best={np.min(costs):.3f} "
                  f"worst={np.max(costs):.3f}")

            test_performances = []
            for w_idx, worker in enumerate(self.workers):
                test_cost = self.evaluate_worker(worker, self.test_env)
                test_performances.append(test_cost)
                worker.update_performance(test_cost)
                hp_writer.writerow([ep + 1, w_idx, worker.hyperparams['actor_lr'],
                                    worker.hyperparams['critic_lr'],
                                    worker.hyperparams['tanh_xplor'], test_cost])
                print(f"  Worker {w_idx}: Val cost={test_cost:.3f}")

                if self.global_best is None or test_cost < self.global_best['val_cost']:
                    self.global_best = {
                        'val_cost': test_cost,
                        'epoch': ep + 1,
                        'worker_id': worker.worker_id,
                        'hyperparams': copy.deepcopy(worker.hyperparams),
                        'learner': copy.deepcopy(worker.learner.state_dict()),
                    }
            hp_file.flush()

            test_mu = float(np.mean(test_performances))
            test_std = float(np.std(test_performances))
            test_gap = (test_mu / self.ref_costs.mean().item() - 1.0) if self.ref_costs is not None else float('nan')
            print(f"\nValidation: mean={test_mu:.3f} ± {test_std:.3f}  best={min(test_performances):.3f}")

            csv_writer.writerow([ep + 1, np.mean(losses), np.mean(costs), np.min(costs), np.max(costs),
                                 np.mean(probs), np.mean(bls), np.mean(norms),
                                 test_mu, test_std, min(test_performances), test_gap])
            csv_file.flush()

            if (ep + 1) % exploit_interval == 0 and ep < n_epochs - 1:
                self.exploit_and_explore(test_performances)

            if (ep + 1) % self.args.checkpoint_period == 0:
                self.save_checkpoint(ep + 1)

        csv_file.close()
        hp_file.close()

        # Final selection: best worker of the final population
        final_perf = [w.last_performance for w in self.workers]
        best_worker = self.workers[int(np.argmin(final_perf))]
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Selected worker (final population): {best_worker.worker_id} "
              f"val cost={best_worker.last_performance:.3f}")
        print(f"Global best seen: worker {self.global_best['worker_id']} at epoch "
              f"{self.global_best['epoch']} val cost={self.global_best['val_cost']:.3f}")
        print(f"{'='*60}")
        return best_worker

    def _worker_record(self, w):
        return {'worker_id': w.worker_id, 'hyperparams': w.hyperparams,
                'last_val_cost': w.last_performance, 'best_val_cost': w.best_performance}

    def save_checkpoint(self, epoch):
        pop_dir = os.path.join(self.args.output_dir, f'population_epoch_{epoch}')
        os.makedirs(pop_dir, exist_ok=True)
        for worker in self.workers:
            torch.save({
                'learner': worker.learner.state_dict(),
                'baseline': worker.baseline.state_dict() if worker.baseline else None,
                'hyperparams': worker.hyperparams,
                'last_val_cost': worker.last_performance,
            }, os.path.join(pop_dir, f'worker_{worker.worker_id}.pt'))
        with open(os.path.join(self.args.output_dir, f'hyperparams_ep{epoch}.json'), 'w') as f:
            json.dump({'epoch': epoch, 'workers': [self._worker_record(w) for w in self.workers]}, f, indent=2)
        print(f"  Population saved to {pop_dir}")

    def save_final_results(self, best_worker):
        best_path = os.path.join(self.args.output_dir, 'best_worker.pt')
        torch.save({
            'learner': best_worker.learner.state_dict(),
            'baseline': best_worker.baseline.state_dict() if best_worker.baseline else None,
            'hyperparams': best_worker.hyperparams,
            'last_val_cost': best_worker.last_performance,
            'worker_id': best_worker.worker_id,
        }, best_path)

        compat_path = os.path.join(self.args.output_dir, f'chkpt_ep{self.args.epoch_count}.pyth')
        torch.save({
            'model': best_worker.learner.state_dict(),
            'learner': best_worker.learner.state_dict(),
            'baseline': best_worker.baseline.state_dict() if best_worker.baseline else None,
            'epoch': self.args.epoch_count,
        }, compat_path)

        summary = {
            'selected_worker_id': best_worker.worker_id,
            'selected_val_cost': best_worker.last_performance,
            'selected_hyperparams': best_worker.hyperparams,
            'global_best': {k: v for k, v in (self.global_best or {}).items() if k != 'learner'},
            'env_kwargs': self.env_kwargs,
            'all_workers_final': [self._worker_record(w) for w in self.workers],
        }
        with open(os.path.join(self.args.output_dir, 'pbt_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}\nFinal Results Saved:")
        print(f"  {best_path}\n  {compat_path}\n  pbt_summary.json, hyperparams_history.csv, loss_gap.csv")
        print(f"{'='*60}")