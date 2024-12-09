from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.utils import *
from marpdan.dep import tqdm

import time
import torch
import os
from torch.utils.data import DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROLLOUTS = 100

# eval_learned_det_pvrp.py
def main(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pb = args.problem_type
    n = args.customers_count 
    m = args.vehicles_count
    
    # Paths
    out_dir = f"./results_{time.strftime('%y%m%d-%H%M')}/{pb}_n{n}m{m}/"
    data_path = f"./data/{pb}_n{n}m{m}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"
    model_path = f"./output/{pb}_n{n}m{m}_241208-1233/chkpt_ep20.pyth"
    
    os.makedirs(out_dir, exist_ok=True)
    print(f" {pb}_n{n}m{m} ".center(96, '-'))

    try:
        # Load data and create dataloader
        data = torch.load(data_path)
        loader = DataLoader(data, batch_size=512)

        # Initialize model
        learner = AttentionLearner(
            cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
            veh_state_size=PVRP_Environment.VEH_STATE_SIZE
        )
        
        # Load weights
        chkpt = torch.load(model_path, map_location="cpu")
        learner.load_state_dict(chkpt["model"])
        learner.to(dev)
        learner.eval()

        with torch.no_grad():
            # Evaluate greedy policy
            evaluate_greedy(learner, loader, data, dev, out_dir)
            
            # Evaluate sampling policy
            evaluate_sampling(learner, loader, data, dev, out_dir, rollouts=100)

    except Exception as e:
        print(f"Error processing {n},{m}: {e}")

def evaluate_greedy(learner, loader, data, device, out_dir):
    learner.greedy = True
    costs, logps = [], []
    
    for batch in tqdm(loader, desc="Evaluating Greedy"):
        batch = batch.to(device)
        env = PVRP_Environment(data, batch)
        
        _, logp, rewards = learner(env)
        costs.append(-torch.stack(rewards).sum(0).squeeze(1))
        logps.append(torch.stack(logp).sum(0).squeeze(1))
    
    costs = torch.cat(costs, 0)
    probs = torch.cat(logps, 0).exp()
    print(f"Greedy: {costs.mean():.3f} ± {costs.std():.3f} w.p. {probs.mean():.3g}")
    torch.save(costs, os.path.join(out_dir, "mardan_greedy.pyth"))

def evaluate_sampling(learner, loader, data, device, out_dir, rollouts=100):
    learner.greedy = False
    costs, logps = [], []
    
    for batch in tqdm(loader, desc="Evaluating Sampling"):
        batch = batch.to(device)
        env = PVRP_Environment(data, batch)
        
        roll_costs, roll_logps = [], []
        for _ in range(rollouts):
            _, logp, rewards = learner(env)
            roll_costs.append(-torch.stack(rewards).sum(0).squeeze(1))
            roll_logps.append(torch.stack(logp).sum(0).squeeze(1))
        
        best_cost, best_idx = torch.stack(roll_costs).min(0, keepdim=True)
        costs.append(best_cost.squeeze(0))
        logps.append(torch.stack(roll_logps).gather(0, best_idx).squeeze(0))
    
    costs = torch.cat(costs, 0)
    probs = torch.cat(logps, 0).exp()
    print(f"Sampling: {costs.mean():.3f} ± {costs.std():.3f} w.p. {probs.mean():.3g}")
    torch.save(costs, os.path.join(out_dir, f"mardan_sample{rollouts}.pyth"))

if __name__ == "__main__":
    main(parse_args())   