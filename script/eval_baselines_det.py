from marpdan.problems import VRP_Dataset, VRPTW_Dataset, VRP_Environment, VRPTW_Environment
from marpdan.externals import lkh_solve, ort_solve
from marpdan.utils import eval_apriori_routes, load_old_weights
from marpdan.dep import tqdm

import torch
import os

for pb in ("cvrp", "cvrptw"):
    for n in (10, 20, 50):
        m = max(n // 5, 1)
        out_dir = "results/{}_n{}m{}/".format(pb, n, m)
        os.makedirs(out_dir, exist_ok = True)
        data_path = "data/{}_n{}m{}/norm_data.pyth".format(pb, n, m)

        print(" {}{} ".format(pb, n).center(96, '-'))

        data = torch.load(data_path)
        nodes = data.nodes.clone()
        nodes[:,:,:2] *= 100
        nodes[:,:,2] *= 200
        if pb == "cvrp":
            unnormed = VRP_Dataset(data.veh_count, 200, 1, nodes[:,:,:3])
            env_data = VRP_Environment(data)
        else:
            nodes[:,:,3:] *= 480
            unnormed = VRPTW_Dataset(data.veh_count, 200, 1, nodes)
            env_data = VRPTW_Environment(data)

        lkh_routes = lkh_solve(unnormed)
        lkh_costs = eval_apriori_routes(env_data, lkh_routes, 1)
        torch.save({"costs": lkh_costs, "routes": lkh_routes}, out_dir + "lkh.pyth")

        ort_routes = ort_solve(unnormed)
        ort_costs = eval_apriori_routes(env_data, ort_routes, 1)
        torch.save({"costs": ort_costs, "routes": ort_routes}, out_dir + "ort.pyth")
