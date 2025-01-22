#!/usr/bin/env python3

from marpdan import *
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.baselines import *
from marpdan.externals import *
from marpdan.dep import *
from marpdan.utils import *
from marpdan.layers import reinforce_loss

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel

import time
import os
from itertools import chain

def train_epoch(args, data, Environment, env_params, bl_wrapped_learner, optim, device, ep):
    bl_wrapped_learner.learner.train()
    
    # Configure DataLoader for parallel processing
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=4,  # Adjust based on CPU cores
        pin_memory=True,
        shuffle=True
    )

    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0

    with tqdm(loader, desc="Ep.#{: >3d}/{: <3d}".format(ep+1, args.epoch_count)) as progress:
        for minibatch in progress:
            if data.cust_mask is None:
                custs, mask = minibatch.to(device), None
            else:
                custs, mask = minibatch[0].to(device), minibatch[1].to(device)

            dyna = Environment(data, custs, mask, *env_params)
            actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
            loss = reinforce_loss(logps, rewards, bl_vals)

            prob = torch.stack(logps).sum(0).exp().mean()
            val = torch.stack(rewards).sum(0).mean()
            bl = bl_vals[0].mean()

            optim.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(
                    chain.from_iterable(grp["params"] for grp in optim.param_groups),
                    args.max_grad_norm
                )
            optim.step()

            progress.set_postfix_str(
                "l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
                    loss, prob, val, bl, grad_norm
                )
            )

            ep_loss += loss.item()
            ep_prob += prob.item()
            ep_val += val.item()
            ep_bl += bl.item()
            ep_norm += grad_norm

    return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm))

def test_epoch(args, test_env, learner, ref_costs):
    learner.eval()
    costs = test_env.nodes.new_zeros(test_env.minibatch_size)
    
    for _ in range(100):
        _, _, rewards = learner(test_env)
        costs -= torch.stack(rewards).sum(0).squeeze(-1)
    costs = costs / 100
    
    mean = costs.mean()
    std = costs.std()
    gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()

    print("Cost on test dataset: {:5.2f} +- {:5.2f} ({:.2%})".format(mean, std, gap))
    return mean.item(), std.item(), gap.item()

def main(args):
    if args.verbose:
        verbose_print = print
    else:
        def verbose_print(*args, **kwargs): pass

    # Initialize distributed training
    if torch.cuda.is_available() and not args.no_cuda:
        if torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.rng_seed is not None:
        torch.manual_seed(args.rng_seed)

    # Generate PVRP data parameters
    gen_params = [
        args.customers_count,
        args.vehicles_count,
        args.veh_capa,
        args.veh_speed,
        args.min_cust_count,
        args.loc_range,
        args.horizon,
        args.spoilage_range
    ]

    # Generate training data
    verbose_print("Generating {} PVRP samples of training data...".format(
        args.iter_count * args.batch_size),
        end=" ", flush=True)
    train_data = PVRP_Dataset.generate(
        args.iter_count * args.batch_size,
        *gen_params
    )
    train_data.normalize()
    verbose_print("Done.")

    # Generate test data
    verbose_print("Generating {} PVRP samples of test data...".format(
        args.test_batch_size),
        end=" ", flush=True)
    test_data = PVRP_Dataset.generate(
        args.test_batch_size,
        *gen_params
    )
    verbose_print("Done.")

    # Get reference solutions
    #21 jan 2025 
    # if ORTOOLS_ENABLED:
    #     ref_routes = ort_solve(test_data)
    # else:
    #     ref_routes = None
    #     print("Warning! No external solver found to compute gaps for test.")
    # test_data.normalize()
    # Get reference solutions with better error handling
    if ORTOOLS_ENABLED:
        print("Computing reference solutions with OR-Tools...")
        try:
            ref_routes = ort_solve(test_data)
            if not any(ref_routes):  # If all routes are empty
                print("Warning: OR-Tools failed to find any valid routes")
                ref_routes = None
        except Exception as e:
            print(f"Error running OR-Tools: {e}")
            ref_routes = None
    else:
        ref_routes = None
        print("Warning! No external solver found to compute gaps for test.")

    test_data.normalize()

    # Continue only if we have valid reference routes or none
    if ref_routes is not None:
        ref_costs = eval_apriori_routes(test_env, ref_routes, 1)
        print(f"Reference cost on test dataset {ref_costs.mean():.2f} +- {ref_costs.std():.2f}")

    # Initialize Environment
    #env_params = [args.spoilage_penalty, args.early_reward, args.unserved_penalty]
    
    env_params = [
    args.spoilage_penalty,
    args.early_reward,
    args.unserved_penalty,
    args.dist_penalty_coef,
    args.pickup_bonus_coef,
    args.idle_penalty_coef,
    args.additional_late_penalty,
    args.capacity_usage_coef
]
    
    test_env = PVRP_Environment(test_data, None, None, *env_params)

    if ref_routes is not None:
        ref_costs = eval_apriori_routes(test_env, ref_routes, 1)
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))
    
    # Move data to device
    test_env.nodes = test_env.nodes.to(device)
    if test_env.init_cust_mask is not None:
        test_env.init_cust_mask = test_env.init_cust_mask.to(device)

    # Initialize model
    verbose_print("Initializing attention model...", end=" ", flush=True)
    learner = AttentionLearner(
        PVRP_Dataset.CUST_FEAT_SIZE,
        PVRP_Environment.VEH_STATE_SIZE,
        args.model_size,
        args.layer_count,
        args.head_count,
        args.ff_size,
        args.tanh_xplor
    )
    
    # Setup parallel processing for model
    if torch.cuda.device_count() > 1:
        learner = DistributedDataParallel(
            learner.to(device),
            device_ids=[local_rank],
            output_device=local_rank
        )
    else:
        learner = learner.to(device)
    verbose_print("Done.")

    # Initialize baseline
    verbose_print("Initializing '{}' baseline...".format(args.baseline_type), end=" ", flush=True)
    if args.baseline_type == "none":
        baseline = NoBaseline(learner)
    elif args.baseline_type == "nearnb":
        baseline = NearestNeighbourBaseline(learner, args.loss_use_cumul)
    elif args.baseline_type == "rollout":
        args.loss_use_cumul = True
        baseline = RolloutBaseline(learner, args.rollout_count, args.rollout_threshold)
    elif args.baseline_type == "critic":
        baseline = CriticBaseline(learner, args.customers_count, args.critic_use_qval, args.loss_use_cumul)
    baseline.to(device)
    verbose_print("Done.")

    # Initialize optimizer and scheduler
    verbose_print("Initializing Adam optimizer...", end=" ", flush=True)
    lr_sched = None
    if args.baseline_type == "critic":
        optim = Adam([
            {"params": learner.parameters(), "lr": args.learning_rate},
            {"params": baseline.parameters(), "lr": args.critic_rate}
        ])
        if args.rate_decay is not None:
            critic_decay = args.rate_decay if args.critic_decay is None else args.critic_decay
            lr_sched = LambdaLR(optim, [
                lambda ep: args.learning_rate * args.rate_decay**ep,
                lambda ep: args.critic_rate * critic_decay**ep
            ])
    else:
        optim = Adam(learner.parameters(), args.learning_rate)
        if args.rate_decay is not None:
            lr_sched = LambdaLR(optim, lambda ep: args.learning_rate * args.rate_decay**ep)
    verbose_print("Done.")

    # Setup checkpointing
    verbose_print("Creating output dir...", end=" ", flush=True)
    args.output_dir = "./output/PVRPn{}m{}_{}".format(
        args.customers_count,
        args.vehicles_count,
        time.strftime("%y%m%d-%H%M")
    ) if args.output_dir is None else args.output_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("'{}' created.".format(args.output_dir))

    if args.resume_state is None:
        start_ep = 0
    else:
        start_ep = load_checkpoint(args, learner, optim, baseline, lr_sched)

    # Training loop
    verbose_print("Running...")
    train_stats = []
    test_stats = []
    try:
        for ep in range(start_ep, args.epoch_count):
            train_stats.append(train_epoch(args, train_data, PVRP_Environment, env_params, baseline, optim, device, ep))
            if ref_routes is not None:
                test_stats.append(test_epoch(args, test_env, learner, ref_costs))

            if args.rate_decay is not None:
                lr_sched.step()

            if (ep+1) % args.checkpoint_period == 0:
                save_checkpoint(args, ep, learner, optim, baseline, lr_sched)

    except KeyboardInterrupt:
        save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
    finally:
        export_train_test_stats(args, start_ep, train_stats, test_stats)

if __name__ == "__main__":
    main(parse_args())