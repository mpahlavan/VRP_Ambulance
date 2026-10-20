#!/usr/bin/env python3
"""
Quick code/paper alignment checks.
Run from the repo root:  python script/check_alignment.py
Each check is independent; a failure in one does not stop the others.
"""
import sys, os, glob, json, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from marpdan import *                                   # AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.baselines._critic import CriticBaseline
from marpdan.utils import parse_args

args = parse_args([])
dev = torch.device("cpu")

# ---------------------------------------------------------------------------
print("\n=== 1. PBT: is the actor the critic wrapper trains the same one the optimizer updates? ===")
try:
    from marpdan.neuroevolution import PBTTrainer
    for k, v in dict(n_workers=2, exploit_interval=5, mini_epochs=1).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    tr = PBTTrainer(args, PVRP_Dataset, PVRP_Environment, AttentionLearner,
                    CriticBaseline, dev, 2)
    w = tr.workers[0]
    same = w.baseline.learner is w.learner
    opt_ids = {id(p) for g in w.optimizer.param_groups for p in g["params"]}
    trained_ids = {id(p) for p in w.baseline.learner.parameters()}
    print("  worker.baseline.learner is worker.learner :", same)
    print("  actor used for training is in optimizer    :", trained_ids <= opt_ids)
    if not same:
        print("  -> PROBLEM: gradients go to a copy the optimizer never steps;"
              " the evaluated actor stays at its random initialization.")
except Exception as e:
    print("  could not run check:", repr(e))

# ---------------------------------------------------------------------------
print("\n=== 2. PBT: does setting learner.C change the policy? ===")
src = inspect.getsource(AttentionLearner._score_customers)
print("  _score_customers uses self.C          :", "self.C" in src)
print("  _score_customers uses self.tanh_xplor :", "self.tanh_xplor" in src)
print("  formula line:", [l.strip() for l in src.splitlines() if "tanh" in l and "=" in l])

# ---------------------------------------------------------------------------
print("\n=== 3. Positional env_params vs PVRP_Environment.__init__ ===")
sig = [p for p in inspect.signature(PVRP_Environment.__init__).parameters][4:]
train_order = ["spoilage_penalty", "unserved_penalty", "dist_penalty_coef",
               "pickup_bonus_coef", "additional_late_penalty",
               "capacity_usage_coef", "idle_penalty_coef"]
for given, received in zip(train_order, sig):
    flag = "" if given == received else "   <-- MISMATCH"
    print(f"  args.{given:<24} lands in  {received}{flag}")

# ---------------------------------------------------------------------------
print("\n=== 4. Values actually used by the training runs (args.json) ===")
keys = ["veh_speed", "spoilage_range", "loc_range", "tanh_xplor", "learning_rate",
        "critic_rate", "spoilage_penalty", "unserved_penalty", "dist_penalty_coef",
        "pickup_bonus_coef", "additional_late_penalty", "capacity_usage_coef",
        "idle_penalty_coef", "checkpoint_period", "baseline_type", "rng_seed"]
for f in sorted(glob.glob("./output/PVRPn10m2_*/args.json")):
    try:
        a = json.load(open(f))
        print(" ", f)
        print("    ", {k: a.get(k) for k in keys})
    except Exception as e:
        print("  ", f, repr(e))

# ---------------------------------------------------------------------------
print("\n=== 5. Scale of the stored test data ===")
for f in sorted(glob.glob("./data/pvrp_n*/norm_data_spoil_*.pyth")):
    try:
        try:
            d = torch.load(f, weights_only=False)
        except TypeError:
            d = torch.load(f)
        s = d.nodes[:, 1:, 3]
        xy = d.nodes[:, :, :2]
        rt = 2 * (d.nodes[:, 1:, :2] - d.nodes[:, :1, :2]).norm(dim=-1) / d.veh_speed
        print(f"  {f}")
        print(f"    veh_speed={d.veh_speed}  coords in [{xy.min():.3f}, {xy.max():.3f}]"
              f"  survival in [{s.min():.4f}, {s.max():.4f}]")
        print(f"    structurally infeasible patients (env rule 2d/f > s): "
              f"{(rt > s).float().mean() * 100:.1f}%")
        # eval scripts give solvers coords*100 and survival*480, then call
        # data.normalize() again before the env evaluates the same routes
        factor = 4.8 * s.max().item() / (xy.max() - xy.min()).item()
        print(f"    deadlines look {factor:.2f}x looser to OR-Tools/GA than to the"
              f" evaluation environment (1.00 would mean consistent)")
    except Exception as e:
        print("  ", f, repr(e))
print()