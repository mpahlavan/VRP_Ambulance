from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.utils import *
from marpdan.dep import tqdm

import torch
import time
import os
from torch.utils.data import DataLoader

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROLLOUTS = 100


def main(args):
    pb = args.problem_type
    n = args.customers_count
    m = args.vehicles_count
    epoch = args.epoch_count
    date = "251008-1229"
    out_pdf_dir = f"./results/{pb}_n{n}m{m}_{time.strftime('%y%m%d-%H%M')}/"
    os.makedirs(out_pdf_dir, exist_ok = True)
    data_path = f"./data/{pb}_n{n}m{m}/norm_data_spoil_{args.spoilage_range[0]}_{args.spoilage_range[1]}.pyth"
    model_path = f"./output/PVRPn{n}m{m}_{date}/chkpt_ep{epoch}.pyth" 

    print(f" {pb}_n{n}m{m} ".center(96, '-'))


    try:
        data = torch.load(data_path)
        loader = DataLoader(data, batch_size=32)

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
            torch.save(costs, out_pdf_dir + "mardan_greedy.pyth")

            # SAMPLING
            learner.greedy = False
            loader = DataLoader(data, batch_size=32)
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
            torch.save(costs, out_pdf_dir + f"mardan_sample{ROLLOUTS}.pyth")

            '''
            beam_width = 5  # Adjust based on computational resources
            costs, logps = [], []
            loader = DataLoader(data, batch_size=512)
            for batch in tqdm(loader, desc="Evaluating Beam Search"):
                batch = batch.to(dev)
                env = PVRP_Environment(data, batch)

                actions, reward, logp = beam_search(env, learner, beam_width)
                costs.append(-reward)
                logps.append(logp.exp())

            costs = torch.tensor(costs)
            probs = torch.tensor(logps)
            print(f"beam {costs.mean():.3f} +- {costs.std():.3f} w.p. {probs.mean():.3g}")
            torch.save(costs, out_dir + "mardan_beam.pyth")
            '''
    except Exception as e:
        print(f"Error processing {n},{m}: {e}")

   
if __name__ == "__main__":
    main(parse_args())     