#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: does best-of-N sampling depend on how the N candidates are batched?

Draws N trajectories for the same instance two ways:
  (a) N separate forward passes with batch size 1   (the original code path)
  (b) one forward pass with the instance replicated N times along the batch dim

If the two cost distributions agree, the batching is neutral and the difference
observed between evaluation runs is just sampling noise. If they are shifted,
there is a real dependency between batch elements that also affects training.
"""

from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.utils import parse_args, load_old_weights

import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_SAMPLES = 100
INSTANCES = [0, 1, 2, 3, 4]      # instances to check
SEED = 12345

COST_PARAMS = {
    'spoilage_penalty': 1,
    'additional_late_penalty': 1,
    'unserved_penalty': 2,
    'idle_penalty_coef': .5,
    'dist_penalty_coef': 0.05,
}


def make_env(batch_data):
    return PVRP_Environment(
        batch_data,
        spoilage_penalty=COST_PARAMS['spoilage_penalty'],
        unserved_penalty=COST_PARAMS['unserved_penalty'],
        additional_late_penalty=COST_PARAMS['additional_late_penalty'],
        dist_penalty_coef=COST_PARAMS['dist_penalty_coef'],
        idle_penalty_coef=COST_PARAMS['idle_penalty_coef'],
    )


def costs_unbatched(data, learner, idx, n):
    """(a) n separate forward passes, batch size 1."""
    nodes = data.nodes[idx].unsqueeze(0).to(dev)
    mask = data.cust_mask[idx:idx+1].to(dev) if data.cust_mask is not None else None

    single = PVRP_Dataset(
        nodes=nodes, cust_mask=mask,
        veh_count=data.veh_count, veh_capa=data.veh_capa, veh_speed=data.veh_speed
    )

    out = []
    learner.greedy = False
    for _ in range(n):
        env = make_env(single)
        with torch.no_grad():
            _, _, rewards = learner(env)
        out.append(-torch.stack(rewards).sum().item())
    return torch.tensor(out)


def costs_batched(data, learner, idx, n):
    """(b) one forward pass with the instance replicated n times."""
    nodes = data.nodes[idx].unsqueeze(0).expand(n, -1, -1).contiguous().to(dev)
    if data.cust_mask is not None:
        mask = data.cust_mask[idx].unsqueeze(0).expand(n, -1).contiguous().to(dev)
    else:
        mask = None

    rep = PVRP_Dataset(
        nodes=nodes, cust_mask=mask,
        veh_count=data.veh_count, veh_capa=data.veh_capa, veh_speed=data.veh_speed
    )

    env = make_env(rep)
    learner.greedy = False
    with torch.no_grad():
        _, _, rewards = learner(env)
    return torch.stack(rewards).sum(0).mul(-1).squeeze(-1).detach().cpu()


def describe(name, c):
    print(f"  {name:<12} mean={c.mean():7.3f}  std={c.std():6.3f}  "
          f"min={c.min():7.3f}  max={c.max():7.3f}")


def main(args):
    pb, n, m = args.problem_type, args.customers_count, args.vehicles_count
    epoch = args.epoch_count
    date = "260908-1016"          # adjust to the checkpoint you want to probe

    data_path = f"./data/{pb}_n{n}m{m}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"
    model_path = f"./output/PVRPn{n}m{m}_{date}/chkpt_ep{epoch}.pyth"

    data = torch.load(data_path)
    data.normalize()

    learner = AttentionLearner(
        cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
        veh_state_size=PVRP_Environment.VEH_STATE_SIZE
    )
    chkpt = torch.load(model_path, map_location="cpu")
    load_old_weights(learner, chkpt["model"])
    learner.to(dev)
    learner.eval()

    print(f"\nComparing batch=1 against batch={N_SAMPLES} sampling, "
          f"{N_SAMPLES} draws per instance\n")

    diffs_mean, diffs_best = [], []

    for idx in INSTANCES:
        torch.manual_seed(SEED + idx)
        c_un = costs_unbatched(data, learner, idx, N_SAMPLES)

        torch.manual_seed(SEED + idx)
        c_ba = costs_batched(data, learner, idx, N_SAMPLES)

        print(f"instance {idx}")
        describe("batch=1", c_un)
        describe(f"batch={N_SAMPLES}", c_ba)

        d_mean = (c_ba.mean() - c_un.mean()).item()
        d_best = (c_ba.min() - c_un.min()).item()
        diffs_mean.append(d_mean)
        diffs_best.append(d_best)
        print(f"  delta mean = {d_mean:+.3f}    delta best-of-N = {d_best:+.3f}\n")

    dm = torch.tensor(diffs_mean)
    db = torch.tensor(diffs_best)
    print("─" * 64)
    print(f"mean shift across instances:       {dm.mean():+.3f}  (std {dm.std():.3f})")
    print(f"best-of-N shift across instances:  {db.mean():+.3f}  (std {db.std():.3f})")
    print("─" * 64)
    print("A mean shift near zero means batching is neutral and the run-to-run")
    print("difference is sampling noise. A consistent nonzero shift means batch")
    print("elements interact, which would also affect training.\n")


if __name__ == "__main__":
    main(parse_args())