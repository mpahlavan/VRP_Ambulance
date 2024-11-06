import json
import random

pb_types = ["vrptw", "svrptw", "sdvrptw", "arp"]
# Counts of patients and ambulances
patient_counts = [10, 20, 50]       # Number of patients
ambulance_counts = [2, 4, 10]       # Number of ambulances
runs = 5
epochs = 20

with open("./cfgs/launch_all.sh", 'w') as sh_f:
    sh_f.write('export MPLBACKEND="Agg"\n\n')
    for pb in pb_types:
        for n_patients, n_ambulances in zip(patient_counts, ambulance_counts):
            for r in range(runs):
                json_fpath = f"./cfgs/{pb}_n{n_patients}m{n_ambulances}_{r}.json"
                with open(json_fpath, 'w') as json_f:
                    json.dump({
                        "rng_seed": random.randint(1e8, 1e9-1),
                        "problem_type": pb,
                        "patient_count": n_patients,
                        "ambulance_count": n_ambulances,
                        "ambulance_capacity": 2,         # Adjust as needed
                        "survival_time_range": [30, 240],# Survival time range in minutes
                        "speed": 1.0,                    # Ambulance speed
                        "gamma": 1.0,                    # Penalty coefficient for late arrivals
                        "sigma": 1.0,                    # Penalty coefficient for early arrivals
                        "pending_cost": 1000,            # Penalty for unserved patients
                        "vacancy_coefficient": 100,      # Penalty coefficient for empty capacities
                        "epoch_count": epochs,
                        "iter_count": 2500,
                        "baseline_type": "critic",
                        "plot_period": epochs,
                        "plot_select": "best"
                    }, json_f, indent=4)
                    json_f.write('\n')
                sh_f.write(f"./script/train.py -f {json_fpath}\n")
            sh_f.write('\n')
