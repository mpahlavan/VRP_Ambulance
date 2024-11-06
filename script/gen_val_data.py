from marpdan.problems import *
from marpdan.externals import lkh_solve, ort_solve
from marpdan.utils import eval_apriori_routes

import torch
from torch.utils.data import DataLoader
import pickle
import os

BATCH_SIZE = 10000
SEED = 231034871114
LATE_PS = [0.05, 0.1, 0.2, 0.3, 0.5]
ROLLOUTS = 100

torch.manual_seed(SEED)

# Define parameters for ARP
PATIENT_COUNTS = [10, 20, 50]        # Number of patients
AMBULANCE_COUNTS = [2, 4, 10]        # Number of ambulances
AMBULANCE_CAPACITY = 2               # Capacity of each ambulance
SURVIVAL_TIME_RANGE = (30, 240)      # Survival time range in minutes
SPEED = 1.0                          # Ambulance speed
GAMMA = 1.0                          # Penalty coefficient for late arrivals
SIGMA = 1.0                          # Penalty coefficient for early arrivals
PENDING_COST = 1000                  # Penalty for unserved patients
VACANCY_COEFFICIENT = 100            # Penalty coefficient for empty capacities

out_dir = "data/"

# CVRP Data
for n_patients, n_ambulances in zip(PATIENT_COUNTS, AMBULANCE_COUNTS):
    problem_dir = os.path.join(out_dir, f"cvrp_n{n_patients}m{n_ambulances}")
    os.makedirs(problem_dir, exist_ok=True)

    data = VRP_Dataset.generate(BATCH_SIZE, n_patients, n_ambulances)

    x_scl = data.nodes[:, :, :2].max() - data.nodes[:, :, :2].min()
    with open(os.path.join(problem_dir, "kool_data.pkl"), 'wb') as f:
        pickle.dump(list(zip(
            data.nodes[:, 0, :2].div(x_scl).tolist(),
            data.nodes[:, 1:, :2].div(x_scl).tolist(),
            data.nodes[:, 1:, 2].tolist(),
            [data.veh_capa for _ in range(BATCH_SIZE)]
        )), f, pickle.HIGHEST_PROTOCOL)

    data.normalize()
    torch.save(data, os.path.join(problem_dir, "norm_data.pyth"))

# CVRPTW Data
for n_patients, n_ambulances in zip(PATIENT_COUNTS, AMBULANCE_COUNTS):
    problem_dir = os.path.join(out_dir, f"cvrptw_n{n_patients}m{n_ambulances}")
    os.makedirs(problem_dir, exist_ok=True)

    data = VRPTW_Dataset.generate(BATCH_SIZE, n_patients, n_ambulances)

    data.normalize()
    torch.save(data, os.path.join(problem_dir, "norm_data.pyth"))

# S-CVRPTW Data (more tw)
for n_patients, n_ambulances in zip(PATIENT_COUNTS, AMBULANCE_COUNTS):
    problem_dir = os.path.join(out_dir, f"s_cvrptw_n{n_patients}m{n_ambulances}")
    os.makedirs(problem_dir, exist_ok=True)

    data = VRPTW_Dataset.generate(BATCH_SIZE, n_patients, n_ambulances, tw_ratio=[0.7, 0.8, 1.0])

    data.normalize()
    torch.save(data, os.path.join(problem_dir, "norm_data.pyth"))

# SD-CVRPTW Data
for n_patients, n_ambulances in zip(PATIENT_COUNTS, AMBULANCE_COUNTS):
    problem_dir = os.path.join(out_dir, f"sd_cvrptw_n{n_patients}m{n_ambulances}")
    os.makedirs(problem_dir, exist_ok=True)

    data = SDVRPTW_Dataset.generate(BATCH_SIZE, n_patients, n_ambulances)
    ort_routes = ort_solve(data)

    data.normalize()
    env = VRPTW_Environment(data)
    ort_costs = eval_apriori_routes(env, ort_routes, 1)

    torch.save(data, os.path.join(problem_dir, "norm_data.pyth"))
    torch.save({
        "costs": ort_costs,
        "routes": ort_routes,
    }, os.path.join(problem_dir, "ort.pyth"))

# Generate data for different problem sizes
for n_patients, n_ambulances in zip(PATIENT_COUNTS, AMBULANCE_COUNTS):
    problem_name = f"arp_n{n_patients}_m{n_ambulances}"
    problem_dir = os.path.join(out_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)

    # Generate ARP dataset
    data = ARP_Dataset.generate(
        batch_size=BATCH_SIZE,
        patient_count=n_patients,
        ambulance_count=n_ambulances,
        ambulance_capacity=AMBULANCE_CAPACITY,
        survival_time_range=SURVIVAL_TIME_RANGE,
        speed=SPEED
    )

    # Normalize data if necessary
    data.normalize()

    # Save the dataset
    data_path = os.path.join(problem_dir, "norm_data.pyth")
    torch.save(data, data_path)

    # Initialize the ARP environment
    env = ARP_Environment(
        data=data,
        gamma=GAMMA,
        sigma=SIGMA,
        pending_cost=PENDING_COST,
        vacancy_coefficient=VACANCY_COEFFICIENT
    )

    # Optionally, generate initial solutions or routes using heuristics or solvers
    # For example, you could use a solver like OR-Tools if available
    # Since ARP is a specialized problem, standard VRP solvers may not directly apply
    # Here, we'll skip route generation and focus on data preparation

    print(f"Generated data for {problem_name} and saved to {problem_dir}")

print("Data generation for ARP completed.")
