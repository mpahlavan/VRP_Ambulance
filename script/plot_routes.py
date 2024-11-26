from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import *
from marpdan.dep import matplotlib as mpl, pyplot as plt
import numpy as np

import torch

SEED = 12348877555
BATCH_SIZE = 128

mpl.rcParams["axes.titlesize"] = 20

# Load trained model
chkpt = torch.load("./output/PVRPn10m2/chkpt_ep20.pyth", map_location="cpu")
learner = AttentionLearner(
    cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
    veh_state_size=PVRP_Environment.VEH_STATE_SIZE
)
learner.load_state_dict(chkpt["model"])
learner.eval()

torch.manual_seed(SEED)

n, m = 10, 2
# Generate PVRP data
data = PVRP_Dataset.generate(BATCH_SIZE, n, m)

# Get reference routes from OR-Tools
ref_routes = ort_solve(data)
x_scl, t_scl = data.normalize()

# Calculate reference costs
# Calculate reference costs
ref_costs = []
for batch_idx, routes in enumerate(ref_routes):
    # Create single instance environment
    single_env = PVRP_Environment(
        data,
        nodes=data.nodes[batch_idx:batch_idx+1],  # Select single instance
        cust_mask=data.cust_mask[batch_idx:batch_idx+1] if data.cust_mask is not None else None
    )
    single_env.reset()
    
    rewards = []
    for route in routes:
        for node in route:
            # Create tensor for single instance [batch=1, 1]
            node_tensor = torch.tensor([[node]], device=data.nodes.device, dtype=torch.long)
            reward = single_env.step(node_tensor)
            rewards.append(reward)
            
    if rewards:  # Check if any rewards were collected
        ref_costs.append(-torch.stack(rewards).sum())
    else:
        ref_costs.append(torch.tensor(float('inf'), device=data.nodes.device))
        
ref_costs = torch.stack(ref_costs)

# Run learned model with full batch
env = PVRP_Environment(data)
with torch.no_grad():
    actions, logps, rewards = learner(env)


costs = torch.stack(rewards).sum(0).mul(-1).squeeze(1)

# Calculate performance gaps
gaps = costs / ref_costs - 1
gaps, sub_idx = gaps.sort()

# Calculate statistics
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

def plot_pvrp_instance(ax, nodes, routes, title):
    """Plot PVRP instance with spoilage times"""
    # Plot depot
    ax.plot(nodes[0,0].item(), nodes[0,1].item(), 'ks', markersize=10, label='Depot')
    
    # Plot customers colored by spoilage time
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

i = 0
for cust, acts, rs, c, ref in zip(data.nodes[sub_idx],
        ([(i[n].item(), j[n].item()) for (i,j) in actions] for n in sub_idx),
        (ref_routes[n] for n in sub_idx),
        costs[sub_idx], ref_costs[sub_idx]):
        
    fig, (ref_ax, ax) = plt.subplots(1, 2, figsize=(20, 8))
    
    g = c / ref - 1
    
    # Plot reference solution
    plot_pvrp_instance(
        ref_ax, 
        cust,
        rs,
        f"OR-Tools (cost = {ref:.3f})"
    )
    
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
    plot_pvrp_instance(
        ax,
        cust,
        learned_routes,
        f"Learned (cost = {c:.3f}, gap = {g:.0%})"
    )
    
    fig.tight_layout()
    fig.savefig(f"results/pvrp_routes_n{n}m{m}_{i:02}_{100*g:.0f}.pdf", 
                bbox_inches='tight')
    i += 1

plt.show()