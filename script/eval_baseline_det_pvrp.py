from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import lkh_solve, ort_solve
from marpdan.utils import * #eval_apriori_routes
from marpdan.dep import tqdm

import torch
import os
import time

def main(args):
    out_dir = f"./results/{args.problem_type}_n{args.customers_count}m{args.vehicles_count}_{time.strftime('%y%m%d-%H%M')}/"
    os.makedirs(out_dir, exist_ok = True)
    data_path = f"./data/{args.problem_type}_n{args.customers_count}m{args.vehicles_count}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"
    
    print(f" {args.problem_type}n{args.customers_count}m{args.vehicles_count} ".center(96, '-'))

    # Load and unnormalize data
    data = torch.load(data_path)
    #torch.save(data, "updated_file.pyth")
    nodes = data.nodes.clone()
    nodes[:,:,:2] *= 100  # Unnormalize coordinates
    nodes[:,:,2] *= 200   # Unnormalize demand (always 1 for PVRP)
    nodes[:,:,3] *= 480   # Unnormalize spoilage times

    # Create unnormalized dataset
    #veh_count,  nodes
    unnormed = PVRP_Dataset(data.veh_count, data.veh_capa, data.veh_speed, nodes)

    env = PVRP_Environment(data)
    
    '''
    # Solve with LKH
    lkh_routes = lkh_solve(unnormed)
    lkh_costs = eval_apriori_routes(env, lkh_routes, 1)
    torch.save({"costs": lkh_costs, "routes": lkh_routes}, out_dir + "lkh.pyth")
    '''

    # Solve with OR-Tools
    ort_routes = ort_solve(unnormed)
    ort_costs = eval_apriori_routes(env, ort_routes, 1)
    torch.save({"costs": ort_costs, "routes": ort_routes}, out_dir + "ort.pyth")


if __name__ == "__main__":
    main(parse_args())  