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
    args = parser.parse_args(argv)
    args.problem_type = "PVRP"
    return args

class PVRPAnalyzer:
    def __init__(self, problem_type, n_customers=10, n_vehicles=2):
        self.problem_type = problem_type
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        # Update the date/time to match your model checkpoint
        self.MODEL_PATH = f"./output/{self.problem_type}n{n_customers}m{n_vehicles}_250123-0903/chkpt_ep10.pyth"
        self.learner = self._load_model()
        
    def _load_model(self):
        try:
            chkpt = torch.load(self.MODEL_PATH, map_location="cpu")
            learner = AttentionLearner(
                cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
                veh_state_size=PVRP_Environment.VEH_STATE_SIZE
            )
            learner.load_state_dict(chkpt["model"])
            learner.eval()
            return learner
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")

    def generate_data(self):
        torch.manual_seed(SEED)
        data = PVRP_Dataset.generate(BATCH_SIZE, self.n_customers, self.n_vehicles)
        ref_routes = ort_solve(data)
        data.normalize()
        return data, ref_routes

    def calculate_reference_costs(self, data, ref_routes):
        ref_costs = []
        for batch_idx, routes in enumerate(ref_routes):
            single_env = PVRP_Environment(
                data,
                nodes=data.nodes[batch_idx:batch_idx+1],
                cust_mask=data.cust_mask[batch_idx:batch_idx+1] if data.cust_mask is not None else None
            )
            single_env.reset()
            
            rewards = []
            for route in routes:
                for node in route:
                    node_tensor = torch.tensor([[node]], device=data.nodes.device, dtype=torch.long)
                    reward = single_env.step(node_tensor)
                    rewards.append(reward)
                    
            if rewards:
                ref_costs.append(-torch.stack(rewards).sum())
            else:
                ref_costs.append(torch.tensor(float('inf'), device=data.nodes.device))
                
        return torch.stack(ref_costs)

    @staticmethod
    def plot_pvrp_instance(ax, nodes, routes, title):
        # Plot depot
        ax.plot(nodes[0,0].item(), nodes[0,1].item(), 'ks', markersize=10, label='Depot')
        
        # Plot customers with spoilage time coloring
        scatter = ax.scatter(nodes[1:,0], nodes[1:,1], 
                           c=nodes[1:,3], 
                           cmap='RdYlGn',
                           label='Customers (color=spoilage time)')
        plt.colorbar(scatter, ax=ax)
        
        # Plot routes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        for route, color in zip(routes, colors):
            route = [0] + route + [0]  # Add depot at start/end
            points = nodes[route]
            ax.plot(points[:,0], points[:,1], '-', 
                   color=color, alpha=0.7, 
                   label=f'Route (len={len(route)-2})')
        
        ax.set_title(title)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def analyze_and_visualize(self):
        # Generate data and get solutions
        data, ref_routes = self.generate_data()
        ref_costs = self.calculate_reference_costs(data, ref_routes)
        
        # Initialize environment and set new_customers attribute
        env = PVRP_Environment(data)
        env.new_customers = False
        
        # Get learned model solutions
        with torch.no_grad():
            actions, _, rewards = self.learner(env)

        # Calculate costs and gaps
        costs = torch.stack(rewards).sum(0).mul(-1).squeeze(1)
        gaps = costs / ref_costs - 1
        gaps, sub_idx = gaps.sort()

        # Print statistics
        fill_ratio = 1-env.vehicles[:,:,2]  # capacity utilization
        print(f"Mean fill ratio on deployed vehicles: {fill_ratio[fill_ratio > 0].mean():.0%}")
        print("      {: ^5} {: ^5} {: ^5} {: ^5} {: ^5}".format("|-", "[", "|", "]", "-|"))
        print("Gaps: {:5.0%} {:5.0%} {:5.0%} {:5.0%} {:5.0%}".format(
            gaps[0], gaps[BATCH_SIZE//4], gaps[BATCH_SIZE//2], gaps[3*BATCH_SIZE//4], gaps[-1]))

        # Select instances to plot
        sub_idx = torch.cat((
            sub_idx[:4],  # Best cases
            sub_idx[BATCH_SIZE//2-4:BATCH_SIZE//2],  # Median cases
            sub_idx[-4:]  # Worst cases
        ))
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Plot selected instances
        for i, (cust, acts, rs, c, ref) in enumerate(zip(
                data.nodes[sub_idx],
                ([(i[n].item(), j[n].item()) for (i,j) in actions] for n in sub_idx),
                (ref_routes[n] for n in sub_idx),
                costs[sub_idx], ref_costs[sub_idx])):
            
            fig, (ref_ax, ax) = plt.subplots(1, 2, figsize=(20, 8))
            g = c / ref - 1
            
            # Plot reference solution
            self.plot_pvrp_instance(ref_ax, cust, rs, f"OR-Tools (cost = {ref:.3f})")
            
            # Convert actions to routes for learned solution
            learned_routes = []
            current_route = []
            for _, node in acts:
                if node == 0:  # Depot
                    if current_route:
                        learned_routes.append(current_route)
                        current_route = []
                else:
                    current_route.append(node)
            if current_route:
                learned_routes.append(current_route)
            
            # Plot learned solution
            self.plot_pvrp_instance(ax, cust, learned_routes,
                                  f"Learned (cost = {c:.3f}, gap = {g:.0%})")
            
            fig.tight_layout()
            fig.savefig(f"results/pvrp_routes_n{self.n_customers}m{self.n_vehicles}_{i:02}_{100*g:.0f}.pdf",
                       bbox_inches='tight')

        plt.show()

def main():
    args = parse_args()
    analyzer = PVRPAnalyzer(args.problem_type, args.customers_count, args.vehicles_count)
    analyzer.analyze_and_visualize()

if __name__ == "__main__":
    main()