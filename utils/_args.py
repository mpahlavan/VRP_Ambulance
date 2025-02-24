from argparse import ArgumentParser
import sys
import json

# Basic configuration defaults
CONFIG_FILE = None
VERBOSE = True
NO_CUDA = True
SEED = None

# PVRP problem defaults
CUST_COUNT = 10
VEH_COUNT = 2
VEH_CAPA = 5
VEH_SPEED = 1
HORIZON = 480
MIN_CUST_COUNT = None
LOC_RANGE = (0, 101)
SPOILAGE_RANGE = (240, 480)

# PVRP reward/penalty coefficients
SPOILAGE_PENALTY = 10.0
EARLY_REWARD = 0.5
UNSERVED_PENALTY = 10.0
DIST_PENALTY_COEF = 1.0
PICKUP_BONUS_COEF = 5.0
IDLE_PENALTY_COEF = 20.0
ADDITIONAL_LATE_PENALTY = 100.0
CAPACITY_USAGE_COEF = 0.4

# Model parameters
MODEL_SIZE = 128
LAYER_COUNT = 3
HEAD_COUNT = 8
FF_SIZE = 512
TANH_XPLOR = 10

# Training parameters
EPOCH_COUNT = 10
ITER_COUNT = 1000
MINIBATCH_SIZE = 512  # Increased for parallel processing
BASE_LR = 0.0001
LR_DECAY = None
MAX_GRAD_NORM = 2
GRAD_NORM_DECAY = None
LOSS_USE_CUMUL = True

# Baseline parameters
BASELINE = "critic"
ROLLOUT_COUNT = 3
ROLLOUT_THRESHOLD = 0.05
CRITIC_USE_QVAL = True
CRITIC_LR = 0.001
CRITIC_DECAY = None

# Testing parameters
TEST_BATCH_SIZE = 512

# Checkpointing parameters
OUTPUT_DIR = None
RESUME_STATE = None
CHECKPOINT_PERIOD = 5

def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

def parse_args(argv=None):
    parser = ArgumentParser()

    # Basic configuration
    parser.add_argument("--config-file", "-f", type=str, default=CONFIG_FILE)
    parser.add_argument("--verbose", "-v", action="store_true", default=VERBOSE)
    parser.add_argument("--no-cuda", action="store_true", default=NO_CUDA)
    parser.add_argument("--rng-seed", type=int, default=SEED)

    # PVRP problem parameters
    group = parser.add_argument_group("PVRP Problem Parameters")
    group.add_argument("--customers-count", "-n", type=int, default=CUST_COUNT,
                      help="Number of customers")
    group.add_argument("--vehicles-count", "-m", type=int, default=VEH_COUNT,
                      help="Number of vehicles")
    group.add_argument("--veh-capa", type=int, default=VEH_CAPA,
                      help="Vehicle capacity")
    group.add_argument("--veh-speed", type=int, default=VEH_SPEED,
                      help="Vehicle speed")
    group.add_argument("--horizon", type=int, default=HORIZON,
                      help="Time horizon (minutes)")
    group.add_argument("--min-cust-count", type=int, default=MIN_CUST_COUNT,
                      help="Minimum number of customers")
    group.add_argument("--loc-range", type=int, nargs=2, default=LOC_RANGE,
                      help="Range for location coordinates")
    group.add_argument("--spoilage-range", type=int, nargs=2, default=SPOILAGE_RANGE,
                      help="Range for spoilage times")

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

    # Model parameters
    group = parser.add_argument_group("Model Parameters")
    group.add_argument("--model-size", "-s", type=int, default=MODEL_SIZE,
                      help="Size of the model")
    group.add_argument("--layer-count", type=int, default=LAYER_COUNT,
                      help="Number of layers")
    group.add_argument("--head-count", type=int, default=HEAD_COUNT,
                      help="Number of attention heads")
    group.add_argument("--ff-size", type=int, default=FF_SIZE,
                      help="Feed-forward network size")
    group.add_argument("--tanh-xplor", type=float, default=TANH_XPLOR,
                      help="Tanh exploration parameter")

    # Training parameters
    group = parser.add_argument_group("Training Parameters")
    group.add_argument("--epoch-count", "-e", type=int, default=EPOCH_COUNT,
                      help="Number of epochs")
    group.add_argument("--iter-count", "-i", type=int, default=ITER_COUNT,
                      help="Number of iterations per epoch")
    group.add_argument("--batch-size", "-b", type=int, default=MINIBATCH_SIZE,
                      help="Batch size")
    group.add_argument("--learning-rate", "-r", type=float, default=BASE_LR,
                      help="Learning rate")
    group.add_argument("--rate-decay", "-d", type=float, default=LR_DECAY,
                      help="Learning rate decay")
    group.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM,
                      help="Maximum gradient norm")
    group.add_argument("--grad-norm-decay", type=float, default=GRAD_NORM_DECAY,
                      help="Gradient norm decay")
    group.add_argument("--loss-use-cumul", action="store_true", default=LOSS_USE_CUMUL,
                      help="Use cumulative loss")

    # Baseline parameters
    group = parser.add_argument_group("Baseline Parameters")
    group.add_argument("--baseline-type", type=str,
                      choices=["none", "nearnb", "rollout", "critic"], default=BASELINE,
                      help="Type of baseline to use")
    group.add_argument("--rollout-count", type=int, default=ROLLOUT_COUNT,
                      help="Number of rollouts")
    group.add_argument("--rollout-threshold", type=float, default=ROLLOUT_THRESHOLD,
                      help="Rollout threshold")
    group.add_argument("--critic-use-qval", action="store_true", default=CRITIC_USE_QVAL,
                      help="Use Q-value in critic")
    group.add_argument("--critic-rate", type=float, default=CRITIC_LR,
                      help="Critic learning rate")
    group.add_argument("--critic-decay", type=float, default=CRITIC_DECAY,
                      help="Critic learning rate decay")

    # Testing parameters
    group = parser.add_argument_group("Testing Parameters")
    group.add_argument("--test-batch-size", type=int, default=TEST_BATCH_SIZE,
                      help="Batch size for testing")

    # Checkpointing parameters
    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--output-dir", "-o", type=str, default=OUTPUT_DIR,
                      help="Output directory")
    group.add_argument("--checkpoint-period", "-c", type=int, default=CHECKPOINT_PERIOD,
                      help="Checkpoint save frequency")
    group.add_argument("--resume-state", type=str, default=RESUME_STATE,
                      help="Path to checkpoint to resume from")

    args = parser.parse_args(argv)
    args.problem_type = "pvrp"  # Fixed to PVRP
    
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))

    return parser.parse_args(argv)