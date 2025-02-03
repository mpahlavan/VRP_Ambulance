# misc.py
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
    """Enhanced route evaluation with device awareness"""
    device = dyna.nodes.device
    mean_cost = torch.zeros(dyna.minibatch_size, device=device)
    
    for _ in range(rollout_count):
        dyna.reset()
        for batch_idx in range(dyna.minibatch_size):
            if batch_idx >= len(routes) or not routes[batch_idx]:
                continue
                
            veh_assignments = [[] for _ in range(dyna.veh_count)]
            # Distribute routes to vehicles
            for vid, route in enumerate(routes[batch_idx]):
                if vid >= dyna.veh_count: break
                veh_assignments[vid] = route + [0]  # Add depot return
                
            # Process vehicle routes
            for vid in range(dyna.veh_count):
                if not veh_assignments[vid]:
                    continue
                    
                # Set vehicle to current route
                cust_idx = torch.tensor([veh_assignments[vid].pop(0)], 
                                      device=device)
                dyna.cur_veh_idx[0] = vid  # Batch size 1
                _ = dyna.step(cust_idx)
                
        # Collect final rewards
        mean_cost -= dyna.cumulative_reward
        
    return mean_cost / rollout_count

def load_old_weights(learner, state_dict):
    """Handle modified architecture when loading weights"""
    try:
        learner.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"Partial weight loading: {e}")
        
    # Handle attention dimension changes
    for layer in learner.cust_encoder.children():
        if hasattr(layer.mha, '_inv_sqrt_d'):
            layer.mha._inv_sqrt_d = layer.mha.key_size_per_head**0.5
            
    for attn in [learner.fleet_attention, learner.veh_attention]:
        if hasattr(attn, '_inv_sqrt_d'):
            attn._inv_sqrt_d = attn.key_size_per_head**0.5

# ... rest of the file remains same