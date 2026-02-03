#!/usr/bin/env python3
#script/train_pbt.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from marpdan import *
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.baselines._critic import CriticBaseline
from marpdan.utils import write_config_file, parse_args, eval_apriori_routes
from marpdan.neuroevolution import PBTTrainer
from marpdan.externals import *
from marpdan.dep import ORTOOLS_ENABLED,tqdm

import torch
import time

def main():
    # Parse arguments
    args = parse_args()
    
    if not hasattr(args, 'critic_use_qval'):
        args.critic_use_qval = False
    if not hasattr(args, 'loss_use_cumul'):
        args.loss_use_cumul = True
    
    # Device
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    if args.rng_seed is not None:
        torch.manual_seed(args.rng_seed)
    
    print(f"{'='*60}")
    print(f"Population-Based Training for Ambulance Routing")
    print(f"{'='*60}")
    print(f"Device: {dev}")
    print(f"Population Size: {args.n_workers}")
    print(f"Exploit Interval: {args.exploit_interval}")
    print(f"Total Epochs: {args.epoch_count}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Problem: n={args.customers_count}, m={args.vehicles_count}")
    print(f"{'='*60}\n")
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = f"./output/PVRPn{args.customers_count}m{args.vehicles_count}_PBT{args.n_workers}_{time.strftime('%y%m%d-%H%M')}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    print(f"Output directory: {args.output_dir}\n")
    
    # Initialize trainer
    print("Initializing PBT Trainer...")
    trainer = PBTTrainer(
        args,
        PVRP_Dataset,
        PVRP_Environment,
        AttentionLearner,
        CriticBaseline,
        dev,
        args.n_workers
    )
    print(" Trainer initialized\n")
    
    # ========== VALIDATION SET (800 samples) ==========
    print(f"Generating {args.test_batch_size} VALIDATION samples...", end=" ", flush=True)
    val_data = PVRP_Dataset.generate(
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
    print("Done.")
    
    # ========== COMPUTE OR-TOOLS REFERENCE (before normalization) ==========
    ref_costs_val = None
    if ORTOOLS_ENABLED:
        print("Computing OR-Tools reference on validation set...", end=" ", flush=True)
        try:
            # Create unnormalized copy for OR-Tools
            val_data_unnorm = PVRP_Dataset(
                val_data.veh_count,
                val_data.veh_capa,
                val_data.veh_speed,
                val_data.nodes.clone()
            )
            # Unnormalize (reverse of normalize())
            val_data_unnorm.nodes[:,:,:2] *= 100  # locations
            val_data_unnorm.nodes[:,:,3] *= 480   # spoilage times
            
            # Solve
            ref_routes_val = ort_solve(val_data_unnorm)
            
            # Normalize validation data for RL
            val_data.normalize()
            
            # Create environment
            env_params = [
                args.spoilage_penalty,
                args.unserved_penalty,
                args.dist_penalty_coef,
                0,  # pickup_bonus_coef (removed)
                args.additional_late_penalty,
                args.capacity_usage_coef,
                args.idle_penalty_coef
            ]
            
            val_env = PVRP_Environment(val_data, None, None, *env_params)
            
            # Evaluate routes
            ref_costs_val = eval_apriori_routes(val_env, ref_routes_val, 1)
            
            print(f"Done. Ref: {ref_costs_val.mean():.2f} ± {ref_costs_val.std():.2f}")
            
            # Set reference for gap computation
            trainer.set_reference_costs(ref_costs_val)
            
        except Exception as e:
            print(f"\n⚠️  Warning: Could not compute OR-Tools reference: {e}")
            val_data.normalize()
    else:
        print("⚠️  OR-Tools not enabled, skipping reference computation")
        val_data.normalize()
    
    # ========== VALIDATION ENVIRONMENT ==========
    print("Creating validation environment...", end=" ", flush=True)
    env_params = [
        args.spoilage_penalty,
        args.unserved_penalty,
        args.dist_penalty_coef,
        0,  # pickup_bonus_coef
        args.additional_late_penalty,
        args.capacity_usage_coef,
        args.idle_penalty_coef
    ]
    
    val_env = PVRP_Environment(val_data, None, None, *env_params)
    val_env.nodes = val_env.nodes.to(dev)
    if hasattr(val_env, 'init_cust_mask') and val_env.init_cust_mask is not None:
        val_env.init_cust_mask = val_env.init_cust_mask.to(dev)
    
    trainer.test_env = val_env
    print("Done.\n")
    
    # ========== TRAIN ==========
    print(f"{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}\n")
    
    try:
        best_worker = trainer.train(
            n_epochs=args.epoch_count,
            exploit_interval=args.exploit_interval,
            mini_epochs=args.mini_epochs
        )
        
        # Save final results
        trainer.save_final_results(best_worker)
        
        print(f"\n{'='*60}")
        print(f"Training completed successfully!")
        print(f"{'='*60}")
        print(f"Best Worker ID: {best_worker.worker_id}")
        print(f"Best Performance: {best_worker.best_performance:.4f}")
        print(f"\nOptimized Hyperparameters:")
        for key, value in best_worker.hyperparams.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # ========== FINAL TEST SET EVALUATION (800 NEW samples) ==========
        print(f"\n{'='*60}")
        print(f"Final Evaluation on TEST SET")
        print(f"{'='*60}\n")
        
        print(f"Generating {args.test_batch_size} TEST samples...", end=" ", flush=True)
        test_data = PVRP_Dataset.generate(
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
        print("Done.")
        
        # Compute OR-Tools reference on test set
        ref_costs_test = None
        if ORTOOLS_ENABLED:
            print("Computing OR-Tools reference on test set...", end=" ", flush=True)
            try:
                test_data_unnorm = PVRP_Dataset(
                    test_data.veh_count,
                    test_data.veh_capa,
                    test_data.veh_speed,
                    test_data.nodes.clone()
                )
                test_data_unnorm.nodes[:,:,:2] *= 100
                test_data_unnorm.nodes[:,:,3] *= 480
                
                ref_routes_test = ort_solve(test_data_unnorm)
                
                test_data.normalize()
                test_env = PVRP_Environment(test_data, None, None, *env_params)
                
                ref_costs_test = eval_apriori_routes(test_env, ref_routes_test, 1)
                print(f"Done. Ref: {ref_costs_test.mean():.2f} ± {ref_costs_test.std():.2f}")
                
            except Exception as e:
                print(f"\n⚠️  Warning: {e}")
                test_data.normalize()
        else:
            test_data.normalize()
        
        # Evaluate best worker on test set
        print("Evaluating best worker on test set...", end=" ", flush=True)
        test_env = PVRP_Environment(test_data, None, None, *env_params)
        test_env.nodes = test_env.nodes.to(dev)
        
        best_worker.learner.eval()
        with torch.no_grad():
            _, _, rewards = best_worker.learner(test_env)
            test_costs = -torch.stack(rewards).sum(dim=0).squeeze(-1)
        
        test_mean = test_costs.mean().item()
        test_std = test_costs.std().item()
        
        if ref_costs_test is not None:
            test_gap = (test_mean / ref_costs_test.mean().item() - 1.0)
            print(f"Done.\n")
            print(f"Test Performance: {test_mean:.2f} ± {test_std:.2f} (Gap: {test_gap:.2%})")
        else:
            print(f"Done.\n")
            print(f"Test Performance: {test_mean:.2f} ± {test_std:.2f}")
        
        # Save test results
        test_results = {
            'test_costs': test_costs.tolist(),
            'test_mean': test_mean,
            'test_std': test_std,
            'ref_costs': ref_costs_test.tolist() if ref_costs_test is not None else None,
            'ref_mean': ref_costs_test.mean().item() if ref_costs_test is not None else None,
            'test_gap': test_gap if ref_costs_test is not None else None
        }
        
        import json
        test_path = os.path.join(args.output_dir, "final_test_results.json")
        with open(test_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"All Results Saved to: {args.output_dir}")
        print(f"  - Training log: loss_gap.csv")
        print(f"  - Best checkpoint: chkpt_ep{args.epoch_count}.pyth")
        print(f"  - PBT summary: pbt_summary.json")
        print(f"  - Test results: final_test_results.json")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current state...")
        
        current_best = max(trainer.workers, key=lambda w: w.get_performance())
        trainer.save_final_results(current_best)
        
        print(f"Current best worker ({current_best.worker_id}) saved to {args.output_dir}")
        print("You can resume training or evaluate this checkpoint.\n")
    
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            current_best = max(trainer.workers, key=lambda w: w.get_performance())
            trainer.save_final_results(current_best)
            print(f"\nEmergency save completed to {args.output_dir}")
        except:
            print("\n❌ Could not save emergency checkpoint")
        
        raise

if __name__ == "__main__":
    main()
