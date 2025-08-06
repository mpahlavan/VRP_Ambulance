from argparse import ArgumentParser
import sys
import json

CONFIG_FILE = None
VERBOSE = True
NO_CUDA = True
SEED = None


PROBLEM = "pvrp"
CUST_COUNT = 10
VEH_COUNT = 2
VEH_CAPA = 5
VEH_SPEED = 2
HORIZON = 480

MIN_CUST_COUNT = None
LOC_RANGE = (0,101)
DEM_RANGE = (5,41)
DUR_RANGE = (10,31)
TW_RATIO = (0.25,0.5,0.75,1.0)
TW_RANGE = (30,91)
DEG_OF_DYN = (0.1,0.25,0.5,0.75)
APPEAR_EARLY_RATIO = (0.0,0.5,0.75,1.0)
SPOILAGE_RANGE = (360,460)

# PVRP reward/penalty coefficients
SPOILAGE_PENALTY = 1
EARLY_REWARD = 0.5
UNSERVED_PENALTY =1
DIST_PENALTY_COEF = 0.05
PICKUP_BONUS_COEF = 1
IDLE_PENALTY_COEF = 10
ADDITIONAL_LATE_PENALTY = 1
CAPACITY_USAGE_COEF = 0
SUCCESS_BONUS = 10.0

# PEND_COST = 2
PEND_GROWTH = None
# LATE_COST = 1
LATE_GROWTH = None
SPEED_VAR = 0.1
LATE_PROB = 0.05
SLOW_DOWN = 0.5
LATE_VAR = 0.2

MODEL_SIZE = 128
LAYER_COUNT = 3
HEAD_COUNT = 8
FF_SIZE = 512
TANH_XPLOR = 11

EPOCH_COUNT = 1000
ITER_COUNT = 100
MINIBATCH_SIZE = 32
BASE_LR = 0.00005
LR_DECAY = None
MAX_GRAD_NORM = 2
GRAD_NORM_DECAY = None
LOSS_USE_CUMUL = True

BASELINE = "critic"
ROLLOUT_COUNT = 3
ROLLOUT_THRESHOLD = 0.05
CRITIC_USE_QVAL = False
CRITIC_LR = 0.0001
CRITIC_DECAY = None
CRITIC_HIDDEN_SIZE = 128 
CRITIC_NUM_LAYERS = 2

TEST_BATCH_SIZE = 800
OUTPUT_DIR = None
RESUME_STATE = None
CHECKPOINT_PERIOD = 5


def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent = 4)


def parse_args(argv = None):
    parser = ArgumentParser()

    parser.add_argument("--config-file", "-f", type = str, default = CONFIG_FILE)
    parser.add_argument("--verbose", "-v", action = "store_true", default = VERBOSE)
    parser.add_argument("--no-cuda", action = "store_true", default = NO_CUDA)
    parser.add_argument("--rng-seed", type = int, default = SEED)

    group = parser.add_argument_group("Data generation parameters")
    group.add_argument("--problem-type", "-p", type=str,
                   choices=["pvrp"], default="pvrp")
#     group.add_argument("--problem-type", "-p", type = str,
#             choices = ["vrp", "vrptw", "svrptw", "sdvrptw", "pvrp"], default = PROBLEM)
    group.add_argument("--customers-count", "-n", type = int, default = CUST_COUNT)
    group.add_argument("--vehicles-count", "-m", type = int, default = VEH_COUNT)
    group.add_argument("--veh-capa", type = int, default = VEH_CAPA)
    group.add_argument("--veh-speed", type = int, default = VEH_SPEED)
    group.add_argument("--horizon", type = int, default = HORIZON)
    group.add_argument("--min-cust-count", type = int, default = MIN_CUST_COUNT)
    group.add_argument("--spoilage-range", type = int, nargs = 2, default = SPOILAGE_RANGE)
    group.add_argument("--loc-range", type = int, nargs = 2, default = LOC_RANGE)
    group.add_argument("--dem-range", type = int, nargs = 2, default = DEM_RANGE)
    group.add_argument("--dur-range", type = int, nargs = 2, default = DUR_RANGE)
    group.add_argument("--tw-ratio", type = float, nargs = '*', default = TW_RATIO)
    group.add_argument("--tw-range", type = int, nargs = 2, default = TW_RANGE)
    group.add_argument("--deg-of-dyna", type = float, nargs = '*', default = DEG_OF_DYN)
    group.add_argument("--appear-early-ratio", type = float, nargs = '*', default = APPEAR_EARLY_RATIO)

    # Standard VRP Environment parameters
    group = parser.add_argument_group("VRP Environment parameters")
    # group.add_argument("--pending-cost", type = float, default = PEND_COST)
    group.add_argument("--pend-cost-growth", type = float, default = PEND_GROWTH)
    # group.add_argument("--late-cost", type = float, default = LATE_COST)
    group.add_argument("--late-cost-growth", type = float, default = LATE_GROWTH)
    group.add_argument("--speed-var", type = float, default = SPEED_VAR)
    group.add_argument("--late-prob", type = float, default = LATE_PROB)
    group.add_argument("--slow-down", type = float, default = SLOW_DOWN)
    group.add_argument("--late-var", type = float, default = LATE_VAR)
    
     # PVRP reward/penalty parameters
    group = parser.add_argument_group("PVRP Reward Parameters")
    group.add_argument("--spoilage-penalty", type=float, default=SPOILAGE_PENALTY,
                      help="Penalty for spoiled goods")
    group.add_argument("--early-reward", type=float, default=EARLY_REWARD,
                      help="Reward for early delivery")
    group.add_argument("--unserved-penalty", type=float, default=UNSERVED_PENALTY,
                      help="Penalty for unserved customers")
    group.add_argument("--dist-penalty-coef", type=float, default=DIST_PENALTY_COEF,
                      help="Coefficient for distance penalty")
    group.add_argument("--pickup-bonus-coef", type=float, default=PICKUP_BONUS_COEF,
                      help="Coefficient for on-time pickup bonus")
    group.add_argument("--idle-penalty-coef", type=float, default=IDLE_PENALTY_COEF,
                      help="Coefficient for idle vehicle penalty")
    group.add_argument("--additional-late-penalty", type=float, default=ADDITIONAL_LATE_PENALTY,
                      help="Additional penalty for late deliveries")
    group.add_argument("--capacity-usage-coef", type=float, default=CAPACITY_USAGE_COEF,
                      help="Coefficient for capacity usage penalty")
    group.add_argument("--success-bonus", type=float, default=SUCCESS_BONUS,
                   help="پاداش زمانی که تمام نودهای feasible بدون دیرکرد به موقع تحویل شوند")

    group = parser.add_argument_group("Model parameters")
    group.add_argument("--model-size", "-s", type = int, default = MODEL_SIZE)
    group.add_argument("--layer-count", type = int, default = LAYER_COUNT)
    group.add_argument("--head-count", type = int, default = HEAD_COUNT)
    group.add_argument("--ff-size", type = int, default = FF_SIZE)
    group.add_argument("--tanh-xplor", type = float, default = TANH_XPLOR)

    group = parser.add_argument_group("Training parameters")
    group.add_argument("--epoch-count", "-e", type = int, default = EPOCH_COUNT)
    group.add_argument("--iter-count", "-i", type = int, default = ITER_COUNT)
    group.add_argument("--batch-size", "-b", type = int, default = MINIBATCH_SIZE)
    group.add_argument("--learning-rate", "-r", type = float, default = BASE_LR)
    group.add_argument("--rate-decay", "-d", type = float, default = LR_DECAY)
    group.add_argument("--max-grad-norm", type = float, default = MAX_GRAD_NORM)
    group.add_argument("--grad-norm-decay", type = float, default = GRAD_NORM_DECAY)
    group.add_argument("--loss-use-cumul", action = "store_true", default = LOSS_USE_CUMUL)

    group = parser.add_argument_group("Baselines parameters")
    group.add_argument("--baseline-type", type = str,
            choices = ["none", "nearnb", "rollout", "critic"], default = BASELINE)
    group.add_argument("--rollout-count", type = int, default = ROLLOUT_COUNT)
    group.add_argument("--rollout-threshold", type = float, default = ROLLOUT_THRESHOLD)
    group.add_argument("--critic-use-qval", action = "store_true", default = CRITIC_USE_QVAL)
    group.add_argument("--critic-rate", type = float, default = CRITIC_LR)
    group.add_argument("--critic-decay", type = float, default = CRITIC_DECAY)

    group = parser.add_argument_group("Testing parameters")
    group.add_argument("--test-batch-size", type = int, default = TEST_BATCH_SIZE)

    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--output-dir", "-o", type = str, default = OUTPUT_DIR)
    group.add_argument("--checkpoint-period", "-c", type = int, default = CHECKPOINT_PERIOD)
    group.add_argument("--resume-state", type = str, default = RESUME_STATE)


    
    
    # Logging parameters
    group = parser.add_argument_group("Logging parameters")
    group.add_argument("--log-dir", type = str, default = None,
                     help="Directory for detailed environment logs")
    group.add_argument("--log-level", type = str, default = "INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                     help="Logging level")

    args = parser.parse_args(argv)
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))

    return parser.parse_args(argv)