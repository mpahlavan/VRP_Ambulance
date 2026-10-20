#!/usr/bin/env python3
import torch
from scipy.stats import ttest_rel, wilcoxon

CRITIC  = "./results/pvrp_n10m2_comprehensive_261006-1217/comprehensive_results.pyth"
ROLLOUT = "./results/pvrp_n10m2_comprehensive_261006-1239/comprehensive_results.pyth"

c = torch.load(CRITIC)["results"]
r = torch.load(ROLLOUT)["results"]

for mode in ["greedy", "sampling"]:
    a = torch.tensor(c[mode]["costs"])
    b = torch.tensor(r[mode]["costs"])
    assert len(a) == len(b)

    d = a - b                      # critic minus rollout
    t, p_t = ttest_rel(a.numpy(), b.numpy())
    w, p_w = wilcoxon(a.numpy(), b.numpy())

    print(f"\n{mode}")
    print(f"  critic   {a.mean():.4f} ± {a.std():.4f}")
    print(f"  rollout  {b.mean():.4f} ± {b.std():.4f}")
    print(f"  paired difference  {d.mean():+.4f} ± {d.std():.4f}")
    print(f"  95% CI  [{d.mean()-1.96*d.std()/len(d)**0.5:+.4f}, "
          f"{d.mean()+1.96*d.std()/len(d)**0.5:+.4f}]")
    print(f"  t-test     p = {p_t:.2e}")
    print(f"  Wilcoxon   p = {p_w:.2e}")
    print(f"  rollout better on {(d > 0).float().mean()*100:.1f}% of instances")