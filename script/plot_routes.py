from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import *
from marpdan.dep import matplotlib as mpl, pyplot as plt
import numpy as np
import torch
import time, os

SEED = 12348877555
BATCH_SIZE = 200
mpl.rcParams["axes.titlesize"] = 20

class PVRPAnalyzer:
    def __init__(self, args):
        self.problem_type = args.problem_type
        self.n_customers = args.customers_count
        self.n_vehicles = args.vehicles_count
        self.veh_capa = args.veh_capa
        self.veh_speed = args.veh_speed
        self.min_cust_count = args.min_cust_count
        self.cust_loc_range = args.loc_range
        self.horizon = args.horizon
        self.spoilage_range = args.spoilage_range
        date = "250419-1031"
        self.MODEL_PATH = f"./output/PVRPn{args.customers_count}m{args.vehicles_count}_{date}/chkpt_ep{args.epoch_count}.pyth"
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
        data = PVRP_Dataset.generate(
            BATCH_SIZE, 
            self.n_customers, 
            self.n_vehicles, 
            self.veh_capa,
            self.veh_speed,
            self.min_cust_count,
            self.cust_loc_range,
            self.horizon,
            self.spoilage_range
        )
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
        ax.plot(nodes[0,0].item(), nodes[0,1].item(), 'ks', markersize=10, label='Depot')
        scatter = ax.scatter(nodes[1:,0], nodes[1:,1], 
                             c=nodes[1:,3], 
                             cmap='RdYlGn',
                             label='Customers (color=spoilage time)')
        plt.colorbar(scatter, ax=ax)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        for route, color in zip(routes, colors):
            route = [0] + route + [0]
            points = nodes[route]
            ax.plot(points[:,0], points[:,1], '-', 
                   color=color, alpha=0.7, 
                   label=f'Route (len={len(route)-2})')
        ax.set_title(title)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def extract_routes_from_env(self, env):
        
        routes = []
        for v in range(env.veh_count):
            indices = (env.vehicle_routes[0] == v).nonzero(as_tuple=True)[0].tolist()
            routes.append(indices)
        return routes

    def analyze_and_visualize(self):
        data, ref_routes = self.generate_data()
        ref_costs = self.calculate_reference_costs(data, ref_routes)
        
        env_batch = PVRP_Environment(data)
        with torch.no_grad():
            actions, _, rewards = self.learner(env_batch)
        init_costs = torch.stack(rewards).sum(0).mul(-1).squeeze(1)
        gaps = init_costs / ref_costs - 1
        gaps, sub_idx = gaps.sort()
        fill_ratio = 1 - env_batch.vehicles[:,:,2]
        print(f"Mean fill ratio on deployed vehicles: {fill_ratio[fill_ratio > 0].mean():.0%}")
        print("      {: ^5} {: ^5} {: ^5} {: ^5} {: ^5}".format("|-", "[", "|", "]", "-|"))
        print("Gaps: {:5.0%} {:5.0%} {:5.0%} {:5.0%} {:5.0%}".format(
            gaps[0], gaps[BATCH_SIZE//4], gaps[BATCH_SIZE//2], gaps[3*BATCH_SIZE//4], gaps[-1]))
        
        sub_idx = torch.cat((sub_idx[:4], sub_idx[BATCH_SIZE//2-4:BATCH_SIZE//2], sub_idx[-4:]))
        output_dir_fig = f"results/pvrp_n{self.n_customers}m{self.n_vehicles}_{time.strftime('%y%m%d-%H%M')}"
        os.makedirs(output_dir_fig, exist_ok=True)
        i = 0
        
        # For each sample in sub_idx
        for idx in sub_idx:
            cust = data.nodes[idx]
            ref_route = ref_routes[idx]
            ref_cost = ref_costs[idx]
            
            # Create a single-instance environment
            # Use a simplified approach for single environment
            single_data = PVRP_Dataset(
                data.veh_count,
                data.veh_capa,
                data.veh_speed,
                data.nodes[idx:idx+1].clone(),
                None if data.cust_mask is None else data.cust_mask[idx:idx+1].clone()
            )
            
            single_env = PVRP_Environment(single_data)
            
            # Run the model on this environment directly
            with torch.no_grad():
                learned_actions, _, learned_rewards = self.learner(single_env)
                
            # Calculate the total cost
            model_cost = -torch.stack(learned_rewards).sum().item()
            gap = model_cost / ref_cost - 1
            
            # Extract routes from learned actions
            learned_routes = []
            for v in range(single_env.veh_count):
                vehicle_nodes = []
                for action_tuple in learned_actions:
                    veh_idx, node_idx = action_tuple
                    if veh_idx[0].item() == v and node_idx[0].item() > 0:  # Skip depot
                        vehicle_nodes.append(node_idx[0].item())
                if vehicle_nodes:
                    learned_routes.append(vehicle_nodes)
            
            # Create visualization
            fig, (ref_ax, ax) = plt.subplots(1, 2, figsize=(20, 8))
            ref_ax.set_title(f"ORTools (cost = {ref_cost:.3f})")
            ax.set_title(f"Learned (cost = {model_cost:.3f}, gap = {gap:.0%})")
            self.plot_pvrp_instance(ref_ax, cust, ref_route, f"ORTools (cost = {ref_cost:.3f})")
            self.plot_pvrp_instance(ax, cust, learned_routes, f"Learned (cost = {model_cost:.3f}, gap = {gap:.0%})")
            
            fig.tight_layout()
            fig.set_size_inches(16,9)
            file_path_fig = f"{output_dir_fig}/+pvrp_routes_n{self.n_customers}m{self.n_vehicles}_{i:02}_{100*gap:.0f}.pdf"
            fig.savefig(file_path_fig, bbox_inches='tight')
            i += 1
            
        plt.show()

def main(args):
    analyzer = PVRPAnalyzer(args)
    analyzer.analyze_and_visualize()

if __name__ == "__main__":
    main(parse_args())
