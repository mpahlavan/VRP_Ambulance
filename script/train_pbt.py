import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from marpdan import *
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.baselines._critic import CriticBaseline
from marpdan.utils import write_config_file, parse_args
from marpdan.neuroevolution import PBTTrainer

import numpy as np
import torch
import time
import json


def env_kwargs_from_args(args):
    """Keyword arguments for PVRP_Environment, matched by name."""
    return dict(
        spoilage_penalty=args.spoilage_penalty,
        unserved_penalty=args.unserved_penalty,
        dist_penalty_coef=args.dist_penalty_coef,
        additional_late_penalty=args.additional_late_penalty,
        idle_penalty_coef=args.idle_penalty_coef,
    )


def make_set(args):
    data = PVRP_Dataset.generate(
        args.test_batch_size,
        args.customers_count,
        args.vehicles_count,
        args.veh_capa,
        args.veh_speed,
        args.min_cust_count,
        args.loc_range,
        args.horizon,
        args.spoilage_range
    )
    data.normalize()
    return data


def main():
    args = parse_args()
    if not hasattr(args, 'critic_use_qval'):
        args.critic_use_qval = False
    if not hasattr(args, 'loss_use_cumul'):
        args.loss_use_cumul = True

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.rng_seed is not None:
        torch.manual_seed(args.rng_seed)
        np.random.seed(args.rng_seed)

    env_kwargs = env_kwargs_from_args(args)

    print(f"{'='*60}\nPopulation-Based Training for Ambulance Routing\n{'='*60}")
    print(f"Device: {dev}")
    print(f"Population Size: {args.n_workers}")
    print(f"Exploit Interval: {args.exploit_interval}")
    print(f"Total Epochs: {args.epoch_count}")
    print(f"Batch Size: {args.batch_size}  Iterations/epoch: {args.iter_count}")
    print(f"Problem: n={args.customers_count}, m={args.vehicles_count}, "
          f"speed={args.veh_speed}, survival={args.spoilage_range}")
    print(f"Environment kwargs: {env_kwargs}")
    print(f"{'='*60}\n")

    if args.output_dir is None:
        args.output_dir = (f"./output/PVRPn{args.customers_count}m{args.vehicles_count}"
                           f"_PBT{args.n_workers}_{time.strftime('%y%m%d-%H%M')}")
    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    print(f"Output directory: {args.output_dir}\n")

    print("Initializing PBT Trainer...")
    trainer = PBTTrainer(args, PVRP_Dataset, PVRP_Environment, AttentionLearner,
                         CriticBaseline, dev, args.n_workers)
    print(" Trainer initialized\n")

    # ========== VALIDATION SET (used for ranking workers) ==========
    print(f"Generating {args.test_batch_size} VALIDATION samples...", end=" ", flush=True)
    val_data = make_set(args)
    val_env = PVRP_Environment(val_data, None, None, **env_kwargs)
    val_env.nodes = val_env.nodes.to(dev)
    if getattr(val_env, 'init_cust_mask', None) is not None:
        val_env.init_cust_mask = val_env.init_cust_mask.to(dev)
    trainer.test_env = val_env
    print("Done.\n")

    print(f"{'='*60}\nStarting Training\n{'='*60}\n")
    try:
        best_worker = trainer.train(
            n_epochs=args.epoch_count,
            exploit_interval=args.exploit_interval,
            mini_epochs=args.mini_epochs
        )
        trainer.save_final_results(best_worker)

        print(f"\nSelected worker: {best_worker.worker_id}")
        print("Selected hyperparameters:")
        for key, value in best_worker.hyperparams.items():
            print(f"  {key}: {value:.6g}")

        # ========== HELD-OUT TEST SET (new samples, not used for selection) ==========
        print(f"\n{'='*60}\nFinal Evaluation on held-out TEST SET\n{'='*60}\n")
        test_data = make_set(args)
        test_env = PVRP_Environment(test_data, None, None, **env_kwargs)
        test_env.nodes = test_env.nodes.to(dev)

        best_worker.learner.eval()
        best_worker.apply_exploration()
        best_worker.learner.greedy = True
        with torch.no_grad():
            _, _, rewards = best_worker.learner(test_env)
            test_costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
        best_worker.learner.greedy = False

        test_mean, test_std = test_costs.mean().item(), test_costs.std().item()
        print(f"Test Performance (greedy): {test_mean:.3f} ± {test_std:.3f}")

        with open(os.path.join(args.output_dir, "final_test_results.json"), 'w') as f:
            json.dump({'test_costs': test_costs.tolist(), 'test_mean': test_mean,
                       'test_std': test_std, 'decoding': 'greedy',
                       'hyperparams': best_worker.hyperparams}, f, indent=2)

        print(f"\nAll results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user! Saving current state...")
        current_best = min(trainer.workers, key=lambda w: w.last_performance)
        trainer.save_final_results(current_best)
        print(f"Current best worker ({current_best.worker_id}) saved to {args.output_dir}")


if __name__ == "__main__":
    main()