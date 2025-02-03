# eval_baseline_det.py
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import eval_apriori_routes
from marpdan.dep import tqdm
import torch
import os

def main(args):
    # Setup output directory
    out_dir = f"./results/pvrp_n{args.customers_count}m{args.vehicles_count}/"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load clustered test data
    data = torch.load(f"./data/pvrp_n{args.customers_count}m{args.vehicles_count}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth")
    
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
    
    # Solve with OR-Tools
    ort_routes = ort_solve(data, args.spoilage_penalty)
    ort_costs = eval_apriori_routes(env, ort_routes, args.rollout_count)
    
    # Save comprehensive results
    torch.save({
        'costs': ort_costs,
        'routes': ort_routes,
        'spoilage_rate': env.spoilage_rate(),
        'capacity_utilization': env.capacity_utilization(),
        'dist_matrix': data.dist_matrix,
        'travel_time_matrix': data.travel_time_matrix
    }, os.path.join(out_dir, "ort_results.pth"))

if __name__ == "__main__":
    # ... (add argument definitions matching train.py)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    main(args)