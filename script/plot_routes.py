# plot_routes.py
from argparse import ArgumentParser
from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import *
from marpdan.dep import matplotlib as mpl, pyplot as plt
import numpy as np
import torch
import os

SEED = 12348877555
BATCH_SIZE = 128
mpl.rcParams["axes.titlesize"] = 20

def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--customers-count", "-n", type=int, default=10)
    parser.add_argument("--vehicles-count", "-m", type=int, default=2)
    parser.add_argument("--spoilage-range", type=int, nargs=2, default=[120, 240])
    parser.add_argument("--epoch-count", "-e", type=int, default=10)
    parser.add_argument("--cluster-prob", type=float, default=0.6)
    args = parser.parse_args(argv)
    args.problem_type = "pvrp"
    return args

class PVRPAnalyzer:
    def __init__(self, args):
        self.n_customers = args.customers_count
        self.n_vehicles = args.vehicles_count
        self.cluster_prob = args.cluster_prob
        self.model_path = f"./output/pvrp_n{self.n_customers}m{self.n_vehicles}_250127-1209/chkpt_ep{args.epoch_count}.pth"
        self.learner = self._load_model()
        
    def _load_model(self):
        learner = AttentionLearner(
            PVRP_Dataset.CUST_FEAT_SIZE,
            PVRP_Environment.VEH_STATE_SIZE
        )
        if os.path.exists(self.model_path):
            learner.load_state_dict(torch.load(self.model_path))
        learner.eval()
        return learner

    def generate_data(self):
        torch.manual_seed(SEED)
        data = PVRP_Dataset.generate(
            BATCH_SIZE, self.n_customers, self.n_vehicles,
            cluster_prob=self.cluster_prob
        )
        data.normalize()
        return data, ort_solve(data)

    def calculate_reference_costs(self, data, ref_routes):
        env = PVRP_Environment(data)
        return eval_apriori_routes(env, ref_routes, 1)

    @staticmethod
    def plot_pvrp_instance(ax, nodes, routes, title, loc_scale, time_scale):
        """Plot with unnormalized coordinates and spoilage times"""
        # Unnormalize data
        depot = nodes[0,:2] * loc_scale
        customers = nodes[1:,:2] * loc_scale
        spoilage_times = nodes[1:,3] * time_scale
        
        ax.plot(depot[0], depot[1], 'ks', markersize=10, label='Depot')
        scatter = ax.scatter(customers[:,0], customers[:,1], 
                           c=spoilage_times, cmap='RdYlGn_r', vmin=0)
        plt.colorbar(scatter, ax=ax, label='Spoilage Time (min)')
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        for route, color in zip(routes, colors):
            path = [0] + route + [0]
            points = np.vstack((depot, customers[route])) if route else depot.reshape(1,-1)
            ax.plot(points[:,0], points[:,1], '-', color=color, alpha=0.7)
        
        ax.set_title(title)
        ax.grid(True)

    def analyze_and_visualize(self):
        data, ref_routes = self.generate_data()
        env = PVRP_Environment(data)
        
        # Get learned solutions
        with torch.no_grad():
            _, _, rewards = self.learner(env)
            learned_cost = -sum(rewards).item()
        
        # Get reference solution
        ref_cost = self.calculate_reference_costs(data, ref_routes).mean().item()
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Unnormalization parameters
        loc_scale = data.nodes[:,:,:2].max()
        time_scale = data.nodes[:,:,3].max()
        
        self.plot_pvrp_instance(ax1, data.nodes[0], ref_routes[0], 
                              f"OR-Tools (Cost: {ref_cost:.1f})", loc_scale, time_scale)
        self.plot_pvrp_instance(ax2, data.nodes[0], self._get_learned_routes(env), 
                              f"Learned (Cost: {learned_cost:.1f})", loc_scale, time_scale)
        
        plt.savefig(f"results/pvrp_comparison_n{self.n_customers}m{self.n_vehicles}.pdf")
        plt.show()

    def _get_learned_routes(self, env):
        routes = [[] for _ in range(env.veh_count)]
        env.reset()
        while not env.done:
            veh_idx = env.cur_veh_idx.item()
            cust_idx = self.learner.step(env)[0].item()
            if cust_idx != 0:
                routes[veh_idx].append(cust_idx)
            env.step(torch.tensor([[cust_idx]]))
        return routes

def main():
    args = parse_args()
    analyzer = PVRPAnalyzer(args)
    analyzer.analyze_and_visualize()

if __name__ == "__main__":
    main()