from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import *
from marpdan.dep import matplotlib as mpl, pyplot as plt
import numpy as np

import torch
import time
import os

SEED = 12348877555
BATCH_SIZE = 128

mpl.rcParams["axes.titlesize"] = 20


class PVRPAnalyzer:
    def __init__(self, args):
        """
        Initialize the PVRP Analyzer.
        
        Args:
            problem_type (str): Type of the problem (e.g., 'PVRP')
            n_customers (int): Number of customers
            n_vehicles (int): Number of vehicles
        """
        self.problem_type = args.problem_type
        self.n_customers = args.customers_count
        self.n_vehicles = args.vehicles_count
        self.veh_capa = args.veh_capa
        self.veh_speed = args.veh_speed
        self.min_cust_count = args.min_cust_count
        self.cust_loc_range =  args.loc_range
        self.horizon = args.horizon
        self.spoilage_range = args.spoilage_range
        date = "241230-1048"
        self.MODEL_PATH  = f"./output/{args.problem_type}n{args.customers_count}m{args.vehicles_count}_{date}/chkpt_ep{args.epoch_count}.pyth"
        self.learner = self._load_model()
        #print( problem_type, n_customers, n_vehicles, epoch)
        
        
    def _load_model(self):
        """Load the trained AttentionLearner model."""
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
        """Generate PVRP dataset and get reference routes."""
        torch.manual_seed(SEED)
        data = PVRP_Dataset.generate(
            BATCH_SIZE, 
            self.n_customers, 
            self.n_vehicles, 
            self.veh_capa,
            self.veh_speed,
            self.min_cust_count,
            self.cust_loc_range,
            self.horizon,
            self.spoilage_range)
        #print(data[0])
        ref_routes = ort_solve(data)
        #print("ref_routes:", ref_routes)
        data.normalize()
        return data, ref_routes

    def calculate_reference_costs(self, data, ref_routes):
        """Calculate costs for reference routes."""
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
        """
        Plot a PVRP instance with routes and spoilage times.
        
        Args:
            ax: Matplotlib axis
            nodes: Node coordinates and features
            routes: List of routes
            title: Plot title
        """
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
        """Perform full analysis and visualization of PVRP solutions."""
        # Generate data and get solutions
        data, ref_routes = self.generate_data()
        ref_costs = self.calculate_reference_costs(data, ref_routes)
        
        # Get learned model solutions
        env = PVRP_Environment(data)
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
            
            # Add some debugging in your code
            print(f"Number of routes in OR-Tools solution:", len(ref_routes[0]))
            print(f"Number of routes in learned solution:", len(learned_routes))

            # Plot learned solution
            self.plot_pvrp_instance(ax, cust, learned_routes,
                                  f"Learned (cost = {c:.3f}, gap = {g:.0%})")
            
            fig.tight_layout()
            
            output_dir_fig = f"results/pvrp_n{self.n_customers}m{self.n_vehicles}_{time.strftime('%y%m%d-%H%M')}"
            os.makedirs(output_dir_fig, exist_ok=True)
            file_path_fig = f"{output_dir_fig}/pvrp_routes_n{self.n_customers}m{self.n_vehicles}_{i:02}_{100*g:.0f}.pdf"
            fig.savefig(file_path_fig, bbox_inches='tight')
            

        plt.show()

def main(args):
    analyzer = PVRPAnalyzer(args)
    analyzer.analyze_and_visualize()
    
    
if __name__ == "__main__":
    main(parse_args())  