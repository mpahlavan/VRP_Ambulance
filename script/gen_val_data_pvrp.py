# gen_val_data.py

from marpdan.problems import *
from marpdan.externals import lkh_solve, ort_solve
from marpdan.dep import *
from marpdan.utils import *

import torch
import os

def main(args):
    out_dir = os.path.join("./data/", f"pvrp_n{args.customers_count}m{args.vehicles_count}")
    os.makedirs(out_dir, exist_ok=True)

    # Generate clustered dataset
    data = PVRP_Dataset.generate(
        batch_size=args.batch_size,
        cust_count=args.customers_count,
        veh_count=args.vehicles_count,
        veh_capa=args.veh_capa,
        veh_speed=args.veh_speed,
        min_cust_count=args.min_cust_count,
        cust_loc_range=args.loc_range,
        spoilage_range=args.spoilage_range,
        cluster_prob=0.6  # Default clustering probability
    )

    # Save raw data with matrices
    raw_path = os.path.join(out_dir, f"raw_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth")
    torch.save(data, raw_path)

    # Normalize and save
    data.normalize()
    norm_path = os.path.join(out_dir, f"norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth")
    torch.save(data, norm_path)

    print(f"Saved dataset with {args.batch_size} instances:")
    print(f"- Raw data: {raw_path}")
    print(f"- Normalized: {norm_path}")

if __name__ == "__main__":
    main(parse_args())