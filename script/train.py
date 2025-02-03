#!/usr/bin/env python3

from marpdan import *
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.baselines import *
from marpdan.externals import *
from marpdan.dep import *
from marpdan.utils import *
from marpdan.layers import reinforce_loss

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import time
import os
from itertools import chain

def train_epoch(args, data, Environment, env_params, bl_wrapped_learner, optim, device, ep):
    bl_wrapped_learner.learner.train()
    loader = DataLoader(data, args.batch_size, True)

    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0
    
    with tqdm(loader, desc=f"Ep.#{ep+1:>3d}/{args.epoch_count:<3d}") as progress:
        for minibatch in progress:
            # Handle data and mask
            if data.cust_mask is not None:
                nodes, mask = minibatch
                nodes = nodes.to(device)
                mask = mask.to(device)
            else:
                nodes = minibatch.to(device)
                mask = None

            # Initialize environment
            dyna = Environment(
                data=data,  # Pass the dataset object
                nodes=nodes,
                cust_mask=mask,
                *env_params
            )
            dyna.nodes = dyna.nodes.to(device)
            
            if mask is not None:
                dyna.init_cust_mask = mask.to(device)

            # Training step
            actions, logps, rewards, bl_vals = bl_wrapped_learner(dyna)
            
            # Process rewards
            if isinstance(rewards, torch.Tensor):
                rewards = [rewards]
                
            loss = reinforce_loss(logps, rewards, bl_vals)
            prob = torch.stack(logps).sum(0).exp().mean()
            val = torch.stack(rewards).sum(0).mean()
            bl = bl_vals[0].mean() if bl_vals else torch.tensor(0)

            # Optimization step
            optim.zero_grad()
            loss.backward()
            
            if args.max_grad_norm:
                grad_norm = clip_grad_norm_(
                    chain.from_iterable(grp["params"] for grp in optim.param_groups),
                    args.max_grad_norm
                )
            
            optim.step()

            # Update progress
            progress.set_postfix_str(
                f"l={loss:.4g} p={prob:9.4g} val={val:6.4g} bl={bl:6.4g} |g|={grad_norm:.4g}"
            )

            ep_loss += loss.item()
            ep_prob += prob.item()
            ep_val += val.item()
            ep_bl += bl.item() if bl else 0
            ep_norm += grad_norm.item() if args.max_grad_norm else 0

    return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm))

def test_epoch(args, test_env, learner, ref_costs):
    learner.eval()
    costs = test_env.nodes.new_zeros(test_env.minibatch_size)
    
    with torch.no_grad():
        for _ in range(100):
            _, _, rewards = learner(test_env)
            costs -= torch.stack(rewards).sum(0).squeeze(-1) if isinstance(rewards, list) else rewards.sum()
            
    costs = costs / 100
    mean, std = costs.mean(), costs.std()
    gap = (costs.to(ref_costs.device) / ref_costs - 1).mean()
    
    print(f"Test cost: {mean:.2f} ± {std:.2f} ({gap:.2%} gap)")
    return mean.item(), std.item(), gap.item()

def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.manual_seed(args.rng_seed if args.rng_seed else int(time.time()))

    # Generate training data with clustering
    train_data = PVRP_Dataset.generate(
        args.iter_count * args.batch_size,
        cust_count=args.customers_count,
        veh_count=args.vehicles_count,
        veh_capa=args.veh_capa,
        veh_speed=args.veh_speed,
        min_cust_count=args.min_cust_count,
        cust_loc_range=args.loc_range,
        horizon=args.horizon,
        spoilage_range=args.spoilage_range,
        cluster_prob=0.6
    )
    train_data.normalize()

    # Generate test data
    test_data = PVRP_Dataset.generate(
        args.test_batch_size,
        cust_count=args.customers_count,
        veh_count=args.vehicles_count,
        veh_capa=args.veh_capa,
        veh_speed=args.veh_speed,
        min_cust_count=args.min_cust_count,
        cust_loc_range=args.loc_range,
        horizon=args.horizon,
        spoilage_range=args.spoilage_range,
        cluster_prob=0.6
    )
    test_data.normalize()

    # Environment parameters
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

    # Initialize environment
    test_env = PVRP_Environment(
        data=test_data,  # Pass dataset instead of individual parameters
        nodes=test_data.nodes,
        cust_mask=test_data.cust_mask,
        *env_params
    )
    test_env.nodes = test_env.nodes.to(dev)

    # Initialize model
    learner = AttentionLearner(
        cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
        veh_state_size=PVRP_Environment.VEH_STATE_SIZE,
        model_size=args.model_size,
        layer_count=args.layer_count,
        head_count=args.head_count,
        ff_size=args.ff_size,
        tanh_xplor=args.tanh_xplor
    ).to(dev)

    # Initialize baseline
    if args.baseline_type == "critic":
        baseline = CriticBaseline(
            learner, 
            args.customers_count,
            args.critic_use_qval,
            args.loss_use_cumul
        )
    elif args.baseline_type == "rollout":
        baseline = RolloutBaseline(
            learner,
            args.rollout_count,
            args.rollout_threshold
        )
    elif args.baseline_type == "nearnb":
        baseline = NearestNeighbourBaseline(
            learner,
            args.loss_use_cumul
        )
    else:
        baseline = NoBaseline(learner)
    baseline.to(dev)

    # Configure optimizer
    optim_groups = [
        {"params": learner.parameters(), "lr": args.learning_rate},
    ]
    if args.baseline_type == "critic":
        optim_groups.append({"params": baseline.parameters(), "lr": args.critic_rate})
        
    optim = Adam(optim_groups)
    lr_sched = LambdaLR(optim, [
        lambda ep: args.learning_rate * (args.rate_decay ** ep),
        lambda ep: args.critic_rate * (args.critic_decay ** ep)
    ]) if args.rate_decay else None

    # Setup output directory
    args.output_dir = args.output_dir or f"./output/PVRPn{args.customers_count}m{args.vehicles_count}_{time.strftime('%y%m%d-%H%M')}"
    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))

    # Training loop
    start_ep = 0
    train_stats = []
    test_stats = []
    
    try:
        for ep in range(start_ep, args.epoch_count):
            train_stats.append(train_epoch(args, train_data, PVRP_Environment, env_params, baseline, optim, dev, ep))
            
            if (ep % args.test_interval) == 0 and test_data is not None:
                test_stats.append(test_epoch(args, test_env, learner, test_data))
                
            if lr_sched:
                lr_sched.step()
                
            if (ep+1) % args.checkpoint_period == 0:
                save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
                
    except KeyboardInterrupt:
        save_checkpoint(args, ep, learner, optim, baseline, lr_sched)
    finally:
        export_train_test_stats(args, start_ep, train_stats, test_stats)

if __name__ == "__main__":
    main(parse_args())