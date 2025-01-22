import torch

import os.path
from itertools import repeat, zip_longest

def actions_to_routes(actions, batch_size, veh_count):
    routes = [[[] for i in range(veh_count)] for b in range(batch_size)]
    for veh_idx, cust_idx in actions:
        for b, (i,j) in enumerate(zip(veh_idx, cust_idx)):
            routes[b][i.item()].append(j.item())
    return routes


def routes_to_string(routes):
    return '\n'.join(
        '0 -> ' + ' -> '.join(str(j) for j in route)
        for route in routes
        )


def export_train_test_stats(args, start_ep, train_stats, test_stats):
    fpath = os.path.join(args.output_dir, "loss_gap.csv")
    with open(fpath, 'a') as f:
        f.write( (' '.join("{: >16}" for _ in range(9)) + '\n').format(
            "#EP", "#LOSS", "#PROB", "#VAL", "#BL", "#NORM", "#TEST_MU", "#TEST_STD", "#TEST_GAP"
            ))
        for ep, (tr,te) in enumerate( zip_longest(train_stats, test_stats, fillvalue=float('nan')), start = start_ep):
            f.write( ("{: >16d}" + ' '.join("{: >16.3g}" for _ in range(8)) + '\n').format(
                ep, *tr, *te))


def _pad_with_zeros(src_it):
    yield from src_it
    yield from repeat(0)


def eval_apriori_routes(dyna, routes, rollout_count):
    """Evaluate routes efficiently with parallel processing"""
    if not routes:
        return dyna.nodes.new_zeros(dyna.minibatch_size)
    
    device = dyna.nodes.device
    batch_size = dyna.minibatch_size
    
    # Compute max route length efficiently
    max_route_len = max(
        max((len(route) + 1 for route in batch_routes), default=1)
        for batch_routes in routes
    )
    
    # Create route tensor in one operation
    route_tensor = torch.zeros(
        (batch_size, dyna.veh_count, max_route_len),
        dtype=torch.long,
        device=device
    )
    
    # Fill route tensor efficiently
    for b, batch_routes in enumerate(routes):
        if batch_routes:
            for v, route in enumerate(batch_routes):
                if route:
                    route_with_depot = route + [0]
                    route_tensor[b, v, :len(route_with_depot)] = torch.tensor(
                        route_with_depot, dtype=torch.long, device=device
                    )
    
    # Evaluate routes in parallel
    costs = torch.zeros(rollout_count, batch_size, device=device)
    
    for roll_idx in range(rollout_count):
        dyna.reset()
        roll_cost = torch.zeros(batch_size, device=device)
        
        # Process all vehicles in parallel for each step
        for step in range(max_route_len):
            for v in range(dyna.veh_count):
                # Get customer indices for current vehicle and step
                # Important: Maintain correct dimensions [batch, 1]
                cust_idx = route_tensor[:, v:v+1, step]  # Shape: [batch, 1]
                reward = dyna.step(cust_idx)
                roll_cost -= reward.squeeze(-1)
        
        costs[roll_idx] = roll_cost
    
    # Compute mean cost across rollouts
    mean_cost = costs.mean(dim=0)
    
    return mean_cost




    # """Evaluate routes for PVRP considering spoilage constraints"""
    # def _pad_with_zeros(route):
    #     """Add depot returns (0) at end of route"""
    #     for node in route:
    #         yield node 
    #     while True:
    #         yield 0

    # # Check if routes is empty or None
    # if not routes:
    #     print("No routes provided")
    #     return dyna.nodes.new_zeros(dyna.minibatch_size)

    # mean_cost = dyna.nodes.new_zeros(dyna.minibatch_size)
    
    # for c in range(rollout_count):
    #     dyna.reset()
        
    #     try:
    #         # Initialize route iterators with proper handling of empty/missing routes
    #         routes_it = []
    #         for batch_idx in range(dyna.minibatch_size):
    #             # Handle case where routes[batch_idx] might not exist
    #             if batch_idx >= len(routes) or not routes[batch_idx]:
    #                 routes_it.append([_pad_with_zeros([])])
    #             else:
    #                 routes_it.append([_pad_with_zeros(route) for route in routes[batch_idx]])
            
    #         rewards = []
    #         while not dyna.done:
    #             try:
    #                 # Get next node for each active vehicle
    #                 cust_idx = []
    #                 for n, i in enumerate(dyna.cur_veh_idx):
    #                     i_val = i.item()
    #                     # Ensure we don't index out of bounds
    #                     if i_val >= len(routes_it[n]):
    #                         cust_idx.append([0])  # Return to depot if no route available
    #                     else:
    #                         cust_idx.append([next(routes_it[n][i_val])])
                    
    #                 # Convert to tensor and step environment
    #                 cust_idx = dyna.nodes.new_tensor(cust_idx, dtype=torch.int64)
    #                 rewards.append(dyna.step(cust_idx))
                    
    #             except Exception as e:
    #                 print(f"Error during route execution: {e}")
    #                 break
            
    #         # Calculate cost even if we broke early
    #         if rewards:
    #             mean_cost += -torch.stack(rewards).sum(dim=0).squeeze(-1)
                
    #     except Exception as e:
    #         print(f"Error during rollout {c}: {e}")
    #         continue
            
    #return mean_cost / rollout_count if rollout_count > 0 else mean_cost


def load_old_weights(learner, state_dict):
    learner.load_state_dict(state_dict)
    for layer in learner.cust_encoder.children():
        layer.mha._inv_sqrt_d = layer.mha.key_size_per_head**0.5
    learner.fleet_attention._inv_sqrt_d = learner.fleet_attention.key_size_per_head**0.5
    learner.veh_attention._inv_sqrt_d = learner.veh_attention.key_size_per_head**0.5

