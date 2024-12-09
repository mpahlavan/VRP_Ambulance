from marpdan.problems import *
from marpdan.externals import lkh_solve, ort_solve
#from marpdan.utils import eval_apriori_routes
from marpdan.dep import *
from marpdan.utils import *

import torch
from torch.utils.data import DataLoader
import pickle
import os


def main(args):
    # Generate data for Perishable VRP
    out_dir = os.path.join("./data/","pvrp_n{}m{}".format(args.customers_count, args.vehicles_count))
    os.makedirs(out_dir, exist_ok=True)

    # Basic dataset with unit demands and spoilage times
    data = PVRP_Dataset.generate(
        batch_size=args.batch_size,
        cust_count=args.customers_count,
        veh_count=args.vehicles_count,
        veh_capa=args.veh_capa,  # Adjusted for unit demands
        veh_speed=args.veh_speed,
        min_cust_count=args.min_cust_count,
        cust_loc_range=args.loc_range,
        spoilage_range= args.spoilage_range #(spoil_min, spoil_max)
    )

    # Save unnormalized version for external solvers
    torch.save(data, os.path.join(out_dir, f"raw_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"))

    # Normalize and save
    data.normalize()
    torch.save(data, os.path.join(out_dir, f"norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"))

    # Get baseline solutions using OR-Tools
    '''
    if ORTOOLS_ENABLED:
        ort_routes = ort_solve(data)
        env = PVRP_Environment(data)
        ort_costs = eval_apriori_routes(env, ort_routes, 1)
        
        torch.save({
            "costs": ort_costs,
            "routes": ort_routes
        }, os.path.join(out_dir, f"ort_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"))
    '''

if __name__ == "__main__":
    main(parse_args())            