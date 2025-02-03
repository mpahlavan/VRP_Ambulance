# eval_learned_det.py
from argparse import ArgumentParser
from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.utils import load_old_weights
from marpdan.dep import tqdm
import torch
import os

def main(args):
    # Setup output directory
    out_dir = f"./results/pvrp_n{args.customers_count}m{args.vehicles_count}_{time.strftime('%y%m%d-%H%M')}/"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load clustered test data
    data_path = f"./data/pvrp_n{args.customers_count}m{args.vehicles_count}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"
    data = torch.load(data_path)
    data.nodes = data.nodes.to(device)
    
    # Initialize environment with current parameters
    env = PVRP_Environment(
        data,
        spoilage_penalty=args.spoilage_penalty,
        early_reward=args.early_reward,
        unserved_penalty=args.unserved_penalty,
        dist_penalty_coef=args.dist_penalty_coef,
        pickup_bonus_coef=args.pickup_bonus_coef,
        idle_penalty_coef=args.idle_penalty_coef,
        additional_late_penalty=args.additional_late_penalty,
        capacity_usage_coef=args.capacity_usage_coef
    )
    
    # Load model with correct state size
    learner = AttentionLearner(
        PVRP_Dataset.CUST_FEAT_SIZE,
        PVRP_Environment.VEH_STATE_SIZE,
        args.model_size,
        args.layer_count,
        args.head_count,
        args.ff_size
    ).to(device)
    
    # Load weights with compatibility handling
    chkpt = torch.load(args.model_path, map_location=device)
    load_old_weights(learner, chkpt["model"])
    learner.eval()
    
    # Evaluation metrics
    metrics = {
        'costs': [],
        'spoilage_rate': [],
        'capacity_utilization': []
    }
    
    with torch.no_grad():
        # Greedy evaluation
        learner.greedy = True
        for batch in tqdm(DataLoader(data, args.test_batch_size), desc="Greedy Eval"):
            batch = batch.to(device)
            env.nodes = batch
            
            _, _, rewards = learner(env)
            if isinstance(rewards, (list, tuple)):
                rewards = torch.stack(rewards).sum(0)
                
            metrics['costs'].append(-rewards.sum())
            metrics['spoilage_rate'].append(env.spoilage_rate())
            metrics['capacity_utilization'].append(env.capacity_utilization())
            
        # Save greedy results
        torch.save({
            'costs': torch.stack(metrics['costs']),
            'spoilage_rate': torch.stack(metrics['spoilage_rate']),
            'capacity_utilization': torch.stack(metrics['capacity_utilization'])
        }, os.path.join(out_dir, "greedy_results.pth"))
        
        # Sampling evaluation
        learner.greedy = False
        metrics = {k: [] for k in metrics.keys()}
        
        for batch in tqdm(DataLoader(data, args.test_batch_size), desc="Sampling Eval"):
            batch = batch.to(device)
            env.nodes = batch
            
            roll_results = []
            for _ in range(args.rollout_count):
                _, _, rewards = learner(env)
                if isinstance(rewards, (list, tuple)):
                    rewards = torch.stack(rewards).sum(0)
                roll_results.append({
                    'cost': -rewards.sum(),
                    'spoilage': env.spoilage_rate(),
                    'utilization': env.capacity_utilization()
                })
                
            # Take best rollout
            best = min(roll_results, key=lambda x: x['cost'])
            for k, v in best.items():
                metrics[k].append(v)
                
        # Save sampling results
        torch.save({
            'costs': torch.stack(metrics['costs']),
            'spoilage_rate': torch.stack(metrics['spoilage_rate']),
            'capacity_utilization': torch.stack(metrics['capacity_utilization'])
        }, os.path.join(out_dir, f"sampling_results_{args.rollout_count}.pth"))

if __name__ == "__main__":
    parser = ArgumentParser()
    # ... (add argument definitions matching train.py)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    main(args)