from marpdan.problems import *
from marpdan.externals import lkh_solve, ort_solve
from marpdan.utils import eval_apriori_routes

import torch
from torch.utils.data import DataLoader
import pickle
import os

BATCH_SIZE = 10000
SEED = 231034871114
SPOILAGE_RANGES = (60, 240) # [(30, 120), (60, 240), (120, 480)]  # Different spoilage time ranges
ROLLOUTS = 100

torch.manual_seed(SEED)

# Generate data for Perishable VRP
for n, m in ((10,2), (20,4)):#, (50,10)
    out_dir = "data/pvrp_n{}m{}".format(n, m)
    os.makedirs(out_dir, exist_ok=True)

    # Generate for different spoilage time ranges
    for spoil_min, spoil_max in SPOILAGE_RANGES:
        # Basic dataset with unit demands and spoilage times
        data = PVRP_Dataset.generate(BATCH_SIZE, n, m)
        
        # Normalize and save
        data.normalize()
        torch.save(data, os.path.join(out_dir, f"norm_data_spoil{spoil_min}_{spoil_max}.pyth"))
        
        # Get baseline solutions using OR-Tools
        ort_routes = ort_solve(data)
        
        env = PVRP_Environment(data)
        ort_costs = eval_apriori_routes(env, ort_routes, 1)

        torch.save({
            "costs": ort_costs,
            "routes": ort_routes
        }, os.path.join(out_dir, f"ort_spoil{spoil_min}{spoil_max}.pyth"))