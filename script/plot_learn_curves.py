# plot_learn_curves.py
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("result_path", help="Path to training stats CSV")
    parser.add_argument("--output-path", "-o", default=None)
    parser.add_argument("--font-size", type=int, default=14)
    return parser.parse_args()

def main(args):
    mpl.rc('font', size=args.font_size)
    results = np.genfromtxt(args.result_path, delimiter=',', names=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main performance plot
    axs[0,0].plot(results['ep'], results['val'], label='Train Reward')
    axs[0,0].plot(results['ep'], results['bl'], label='Baseline')
    axs[0,0].plot(results['ep'], -results['test_mu'], label='Test Reward')
    axs[0,0].set_ylabel("Reward")
    axs[0,0].legend()
    
    # Secondary metrics
    axs[0,1].plot(results['ep'], results['prob'])
    axs[0,1].set_ylabel("Action Probability")
    
    axs[1,0].plot(results['ep'], results['loss'])
    axs[1,0].set_ylabel("Loss")
    
    axs[1,1].plot(results['ep'], results['norm'])
    axs[1,1].set_ylabel("Grad Norm")
    
    plt.savefig(args.output_path or args.result_path.replace('.csv', '.pdf'))
    plt.show()

if __name__ == "__main__":
    main(parse_args())