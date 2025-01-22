from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.utils import *
from marpdan.dep import tqdm

import copy
import torch
from torch.utils.data import DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROLLOUTS = 100

def beam_search(env, learner, beam_width=5, max_steps=50):
    """
    Optimized beam search implementation
    """
    env.reset()
    if env.new_customers:
        learner._encode_customers(env.nodes, env.cust_mask)
    
    device = env.nodes.device
    initial_state = {
        "vehicles": env.vehicles.clone(),
        "veh_done": env.veh_done.clone(),
        "served": env.served.clone(),
        "mask": env.mask.clone(),
        "cur_veh_idx": env.cur_veh_idx.clone(),
        "vehicle_routes": env.vehicle_routes.clone()
    }
    
    beams = [([], initial_state, 0.0, 0.0)]  # actions, state_dict, cum_reward, cum_logp
    step_count = 0
    
    while step_count < max_steps:
        step_count += 1
        candidates = []
        
        for actions, state, cum_reward, cum_logp in beams:
            # Load state into environment
            env.load_state_dict(state)
            if env.done:
                candidates.append((actions, state, cum_reward, cum_logp))
                continue

            # Get next actions
            veh_repr = learner._repr_vehicle(env.vehicles, env.cur_veh_idx, env.mask)
            compat = learner._score_customers(veh_repr)
            logp = learner._get_logp(compat, env.cur_veh_mask)
            
            # Get top-k actions
            topk_values, topk_indices = logp.topk(min(beam_width, logp.size(-1)))
            
            for value, idx in zip(topk_values.squeeze(), topk_indices.squeeze()):
                cust_idx = torch.tensor([[idx.item()]], device=device, dtype=torch.long)
                reward = env.step(cust_idx)
                
                # Save new state
                new_state = env.state_dict()
                
                candidates.append((
                    actions + [cust_idx],
                    new_state,
                    cum_reward + reward.item(),
                    cum_logp + value.item()
                ))
                
                # Restore original state for next iteration
                env.load_state_dict(state)
        
        # Check if all candidates are done
        if all(env.load_state_dict(state) or env.done for _, state, _, _ in candidates):
            best_beam = max(candidates, key=lambda x: x[2])
            return best_beam[2], best_beam[3]
        
        # Select top-k beams
        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_width]
    
    # Return best solution if max steps reached
    best_beam = max(beams, key=lambda x: x[2])
    return best_beam[2], best_beam[3]
def main(args):
    pb = args.problem_type
    n = args.customers_count 
    m = args.vehicles_count
    epoch = args.epoch_count
    out_dir = f"./results/{pb}_n{n}m{m}/"
    data_path = f"./data/{pb}_n{n}m{m}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"
    model_path = f"./output/{pb}n{n}m{m}/chkpt_ep{epoch}.pyth" #f"./output/{pb}_n{n}m{m}.pyth"

    print(f" {pb}{n} ".center(96, '-'))

    try:
        data = torch.load(data_path)
        loader = DataLoader(data, batch_size=512)

        # Initialize learner with PVRP feature sizes
        learner = AttentionLearner(
            cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,  # 4: x,y,demand,spoilage
            veh_state_size=PVRP_Environment.VEH_STATE_SIZE  # 4: x,y,capacity,time
        )
        chkpt = torch.load(model_path, map_location="cpu")
        load_old_weights(learner, chkpt["model"])
        learner.to(dev)
        learner.eval()

        with torch.no_grad():
            '''
            # GREEDY
            learner.greedy = True
            costs = []
            logps = []
            for batch in tqdm(loader, desc="Evaluating Greedy"):
                batch = batch.to(dev)
                env = PVRP_Environment(data, batch)

                _, logp, rewards = learner(env)
                costs.append(-torch.stack(rewards).sum(0).squeeze(1))
                logps.append(torch.stack(logp).sum(0).squeeze(1))
            
            costs = torch.cat(costs, 0)
            probs = torch.cat(logps, 0).exp()
            print(f"greedy {costs.mean():.3f} +- {costs.std():.3f} w.p. {probs.mean():.3g}")
            torch.save(costs, out_dir + "mardan_greedy.pyth")

            # SAMPLING
            learner.greedy = False
            loader = DataLoader(data, batch_size=512)
            costs = []
            logps = []
            for batch in tqdm(loader, desc="Evaluating Sampling"):
                batch = batch.to(dev)
                env = PVRP_Environment(data, batch)

                roll_costs = []
                roll_logps = []
                for _ in range(ROLLOUTS):
                    _, logp, rewards = learner(env)
                    roll_costs.append(-torch.stack(rewards).sum(0).squeeze(1))
                    roll_logps.append(torch.stack(logp).sum(0).squeeze(1))
                
                best_cost, best_idx = torch.stack(roll_costs).min(0, keepdim=True)
                costs.append(best_cost.squeeze(0))
                logps.append(torch.stack(roll_logps).gather(0, best_idx).squeeze(0))
            
            costs = torch.cat(costs, 0)
            probs = torch.cat(logps, 0).exp()
            print(f"sample {costs.mean():.3f} +- {costs.std():.3f} w.p. {probs.mean():.3g}")
            torch.save(costs, out_dir + f"mardan_sample{ROLLOUTS}.pyth")
            '''
            # BEAM SEARCH
            print("\nStarting Beam Search Evaluation...")
            beam_width = 5
            costs = []
            logps = []
            
            for batch_idx, batch in enumerate(loader):
                batch_size = batch.size(0)
                print(f"\nProcessing batch {batch_idx+1}/{len(loader)} ({batch_size} instances)")
                
                for i in range(batch_size):
                    instance = batch[i:i+1].to(dev)
                    env = PVRP_Environment(data, instance)
                    
                    try:
                        cost, logp = beam_search(env, learner, beam_width)
                        costs.append(cost)
                        logps.append(logp)
                        if (len(costs) % 10) == 0:
                            print(f"Processed {len(costs)} instances. Current mean cost: {sum(costs)/len(costs):.3f}")
                    except Exception as e:
                        print(f"Error in instance {batch_idx * batch.size(0) + i}: {e}")
                        continue
            
            if costs:
                costs = torch.tensor(costs, device=dev)
                logps = torch.tensor(logps, device=dev)
                print(f"\nBeam search results (width={beam_width}):")
                print(f"Cost: {costs.mean():.3f} +- {costs.std():.3f}")
                print(f"Probability: {logps.exp().mean():.3g}")
                torch.save(costs, out_dir + f"mardan_beam{beam_width}.pyth")


    except Exception as e:
        print(f"Error processing {n},{m}: {e}")

   
if __name__ == "__main__":
    main(parse_args())     
