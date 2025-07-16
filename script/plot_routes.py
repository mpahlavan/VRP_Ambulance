#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot OR-Tools vs. learned routes for PVRP.
"""

import os, time, numpy as np, torch
from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import parse_args
from marpdan.dep import matplotlib as mpl, pyplot as plt

# ---------------- global style ----------------
mpl.rcParams["axes.titlesize"] = 20
SEED       = 12348877555
BATCH_SIZE =1600


# ======================================================================
#  helper – compute late sets
# ======================================================================
def compute_late_sets(env):
    """
    Extract two sets of customer indices from a finished PVRP_Environment.

    Returns
    -------
    late_nodes          : set[int]  – late arrival at the node itself
    late_depot_nodes    : set[int]  – picked on-time, but package reached depot late
    """
    # ---------- guard: make sure env looks finished ----------
    if not hasattr(env, "vehicles"):
        # env.reset() has probably not been called, or wrong object was passed
        return set(), set()

    # ---------- A) late at node ----------
    late_nodes = set()
    if getattr(env, "late_nodes", None) is not None:
        late_nodes = set(
            torch.nonzero(env.late_nodes[0], as_tuple=False).flatten().tolist()
        )
        late_nodes.discard(0)                # remove depot if present

    # ---------- B) late at depot (check each node) ----------
    late_depot = set()
    arrival_in_depot = env.vehicles[0, :, 3]     # [veh] depot arrival time
    spoilage_times   = env.nodes[0, :, 3]        # [nodes] spoilage times

    for v in range(env.veh_count):
        idxs = torch.nonzero(
            env.vehicle_routes[0] == v, as_tuple=False
        ).flatten()
        if idxs.numel() == 0:
            continue
        t_dep = arrival_in_depot[v].item()
        for n in idxs.tolist():
            if n != 0 and t_dep > spoilage_times[n]:
                late_depot.add(n)

    return late_nodes, late_depot


# ======================================================================
#  helper – plot one instance
# ======================================================================
def plot_instance(ax, nodes, routes, title="",
                  late_nodes=None, late_depot_nodes=None, cmap="RdYlGn"):
    """Visualise a single instance on axis ax."""
    late_nodes       = late_nodes or set()
    late_depot_nodes = late_depot_nodes or set()

    # depot
    ax.plot(nodes[0, 0], nodes[0, 1], "ks", ms=9, label="Depot")

    # customers coloured by spoilage
    sc = ax.scatter(nodes[1:, 0], nodes[1:, 1],
                    c=nodes[1:, 3], cmap=cmap,
                    edgecolors="k", linewidths=0.4,
                    label="Customers (spoilage)")
    plt.colorbar(sc, ax=ax, shrink=0.8)

    # routes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    for rt, color in zip(routes, colors):
        pts = nodes[[0] + rt + [0]]
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=0.75, linewidth=2)

    # highlight late-at-node (filled red)
    for n in late_nodes:
        ax.plot(nodes[n, 0], nodes[n, 1],
                "ro", ms=9,
                label="Late at node" if n == min(late_nodes) else "")

    # highlight late-at-depot (open red)
    for n in late_depot_nodes:
        ax.plot(nodes[n, 0], nodes[n, 1],
                marker="o", ms=11,
                markerfacecolor="none",
                markeredgecolor="red", markeredgewidth=1.8,
                label="Late at depot" if n == min(late_depot_nodes) else "")

    # اضافه کردن annotation برای spoilage times
    for i in range(1, len(nodes)):
        spoilage = nodes[i, 3].item()
        ax.annotate(f"{spoilage:.2f}", (nodes[i, 0].item(), nodes[i, 1].item()), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8, frameon=True)


# ======================================================================
#  Analyzer class
# ======================================================================
class PVRPAnalyzer:
    def __init__(self, args):
        self.args = args
        self.nC, self.nV = args.customers_count, args.vehicles_count
        self.veh_capa    = args.veh_capa
        self.veh_speed   = args.veh_speed
        self.min_cust    = args.min_cust_count
        self.loc_rng     = args.loc_range
        self.horizon     = args.horizon
        self.spoil_rng   = args.spoilage_range
        self.epoch_num   = args.epoch_count

        # env-params
        self.env_params = [
            args.spoilage_penalty,
            args.unserved_penalty,
            args.pickup_bonus_coef,
            args.additional_late_penalty,
            args.capacity_usage_coef,
            args.dist_penalty_coef,
            args.idle_penalty_coef,
            args.success_bonus,
        ]

        # load model
        date = "250714-1003"   # adjust to your folder
        self.model_pth = (
            f"./output/PVRPn{self.nC}m{self.nV}_{date}/"
            f"chkpt_ep{self.epoch_num}.pyth"
        )
        self.learner = self._load_model()

    def _load_model(self):
        chkpt = torch.load(self.model_pth, map_location="cpu")
        net   = AttentionLearner(
            cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
            veh_state_size=PVRP_Environment.VEH_STATE_SIZE,
        )
        net.load_state_dict(chkpt["model"])
        net.eval()
        return net

    # ----------------------------------------------------
    def generate_data(self):
        torch.manual_seed(SEED)
        data = PVRP_Dataset.generate(
            BATCH_SIZE, self.nC, self.nV, self.veh_capa, self.veh_speed,
            self.min_cust, self.loc_rng, self.horizon, self.spoil_rng
        )
        ref_routes = ort_solve(data)
        data.normalize()
        return data, ref_routes

    # ----------------------------------------------------
    def calc_ref_costs(self, data, ref_routes):
        ref_costs = []
        served = late = unserved = 0

        for b, routes in enumerate(ref_routes):
            sd = PVRP_Dataset(
                data.veh_count, data.veh_capa, data.veh_speed,
                data.nodes[b:b+1].clone(),
                None if data.cust_mask is None else data.cust_mask[b:b+1].clone()
            )
            env = PVRP_Environment(sd, None, None, *self.env_params)
            env.reset()

            rs = []
            for rt in routes:
                for node in rt:
                    rs.append(env.step(torch.tensor([[node]], dtype=torch.long)))
            if not env.done:
                rs.append(env.step(torch.tensor([[0]], dtype=torch.long)))
            ref_costs.append(-torch.stack(rs).sum())

            served  += env.served.sum().item()
            late    += env.late_nodes.sum().item()
            if env.init_cust_mask is not None:
                eff = ~env.init_cust_mask.bool()
                unserved += (eff & ~env.served & ~env.infeasible_nodes).sum().item()
            else:
                unserved += (~env.served & ~env.infeasible_nodes).sum().item()

        n = len(ref_routes)
        print(f"\nORTools reference: Served={served/n:.2f} Late={late/n:.2f} "
              f"Unserved={unserved/n:.2f}")
        return torch.stack(ref_costs)

    # ----------------------------------------------------
    def extract_routes_from_actions(self, actions):
        """
        استخراج مسیرها از actions مدل یادگیری به شکل مشابه کد اولیه
        """
        learned_routes = [[] for _ in range(self.nV)]
        
        for action in actions:
            if isinstance(action, tuple) and len(action) == 2:
                veh_idx, node_idx = action
                v = veh_idx[0].item() if hasattr(veh_idx[0], 'item') else veh_idx[0]
                n = node_idx[0].item() if hasattr(node_idx[0], 'item') else node_idx[0]
                if n > 0:  # به جز دپو
                    learned_routes[v].append(n)
            elif hasattr(action, 'item'):
                # اگر action یک تنسور تک بعدی است
                n = action.item()
                if n > 0:
                    # باید vehicle index را از محیط دریافت کنیم
                    # این بخش ممکن است نیاز به تطبیق با ساختار دقیق actions داشته باشد
                    learned_routes[0].append(n)
        
        # حذف مسیرهای خالی
        learned_routes = [route for route in learned_routes if route]
        return learned_routes

    # ----------------------------------------------------
    def analyze_and_visualize(self):
        data, ref_routes = self.generate_data()
        ref_costs        = self.calc_ref_costs(data, ref_routes)

        # ---------- run learned ----------
        learned_costs = []
        learned_routes_cache = []   # ذخیره مسیرهای یادگیری شده برای plotting
        env_cache = []   # keep envs for later plotting
        
        for b in range(BATCH_SIZE):
            sd  = PVRP_Dataset(
                data.veh_count, data.veh_capa, data.veh_speed,
                data.nodes[b:b+1].clone(),
                None if data.cust_mask is None else data.cust_mask[b:b+1].clone()
            )
            env = PVRP_Environment(sd, None, None, *self.env_params)
            env.reset()
            
            with torch.no_grad():
                learned_actions, _, rs = self.learner(env)
            
            if not env.done:
                rs.append(env.step(torch.tensor([[0]], dtype=torch.long)))
            
            learned_costs.append(-torch.stack(rs).sum())
            
            # استخراج مسیرهای یادگیری شده
            learned_routes = self.extract_routes_from_actions(learned_actions)
            learned_routes_cache.append(learned_routes)
            
            env_cache.append(env)   # store for plotting
            
        learned_costs = torch.stack(learned_costs)

        gaps = learned_costs / ref_costs - 1
        print(f"\nAvg ORTools cost {ref_costs.mean():.1f}")
        print(f"Avg learned cost {learned_costs.mean():.1f}")
        print(f"Gap mean {gaps.mean():+.1%} (max {gaps.max():+.1%} min {gaps.min():+.1%})")

        # ---------- pick samples ----------
        order = gaps.argsort()
        picks = torch.cat([order[:4],
                           order[BATCH_SIZE//2-2:BATCH_SIZE//2+2],
                           order[-4:]]).tolist()

        # ---------- output dir ----------
        now    = time.strftime("%y%m%d-%H%M")
        outdir = f"results/pvrp_n{self.nC}m{self.nV}_{now}_ep{self.epoch_num}"
        os.makedirs(outdir, exist_ok=True)

        # ---------- plot ----------
        for k, idx in enumerate(picks):
            cust  = data.nodes[idx]
            ref_rt = ref_routes[idx]
            learned_rt = learned_routes_cache[idx]
            env_learned = env_cache[idx]

            gap = gaps[idx].item()

            # compute late sets for reference
            # برای مرجع، محیط جدید ایجاد کنیم
            ref_env = PVRP_Environment(
                PVRP_Dataset(
                    data.veh_count, data.veh_capa, data.veh_speed,
                    data.nodes[idx:idx+1].clone(),
                    None if data.cust_mask is None else data.cust_mask[idx:idx+1].clone()
                ),
                None, None, *self.env_params
            )
            ref_env.reset()
            for rt in ref_rt:
                for node in rt:
                    ref_env.step(torch.tensor([[node]], dtype=torch.long))
            if not ref_env.done:
                ref_env.step(torch.tensor([[0]], dtype=torch.long))
            
            late_ref, late_dep_ref = compute_late_sets(ref_env)
            late_lea, late_dep_lea = compute_late_sets(env_learned)

            # محاسبه آمار اضافی
            ref_served = set([node for route in ref_rt for node in route])
            learned_served = set([node for route in learned_rt for node in route])
            all_nodes = set(range(1, len(cust)))  # به جز دپو (0)
            
            ref_unserved = all_nodes - ref_served
            learned_unserved = all_nodes - learned_served

            # plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # عناوین با اطلاعات اضافی
            ref_title = (f"ORTools (cost={ref_costs[idx]:.1f})\n"
                        f"Served: {len(ref_served)}, Late: {len(late_ref)}, "
                        f"Late at depot: {len(late_dep_ref)}, Unserved: {len(ref_unserved)}")
            
            learned_title = (f"Learned (cost={learned_costs[idx]:.1f}, gap={gap:+.0%})\n"
                           f"Served: {len(learned_served)}, Late: {len(late_lea)}, "
                           f"Late at depot: {len(late_dep_lea)}, Unserved: {len(learned_unserved)}")
            
            plot_instance(ax1, cust, ref_rt,
                          title=ref_title,
                          late_nodes=late_ref, late_depot_nodes=late_dep_ref)
            plot_instance(ax2, cust, learned_rt,
                          title=learned_title,
                          late_nodes=late_lea, late_depot_nodes=late_dep_lea)

            # اضافه کردن گره‌های خدمت داده نشده به عنوان علامت‌های 'X'
            for node in ref_unserved:
                ax1.plot(cust[node, 0].item(), cust[node, 1].item(), 'rx', markersize=10, 
                        label="Unserved" if node == min(ref_unserved) else "")
            
            for node in learned_unserved:
                ax2.plot(cust[node, 0].item(), cust[node, 1].item(), 'rx', markersize=10, 
                        label="Unserved" if node == min(learned_unserved) else "")

            fig.tight_layout()
            pdf = f"pvrp_routes_n{self.nC}m{self.nV}_{k:02d}_{100*gap:+.0f}.pdf"
            fig.savefig(os.path.join(outdir, pdf), bbox_inches="tight")
            plt.close(fig)

        print(f"\nPlots saved to: {outdir}")


# ======================================================================
def main(args):
    PVRPAnalyzer(args).analyze_and_visualize()

if __name__ == "__main__":
    main(parse_args())