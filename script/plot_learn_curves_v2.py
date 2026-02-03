#!/usr/bin/env python3
"""
Flexible learning curve plotter for PVRP training
Works with any baseline: critic, rollout, nearest neighbor,hybrid or none

Auto-reads baseline type from args.json if available
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
from argparse import ArgumentParser
import os
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("result_path",
            help="Path to .csv file containing results and stats")
    parser.add_argument("--output-path", "-o", default=None,
            help="Path to the pdf file the plot will be exported to")
    parser.add_argument("--font-size", default=20, type=int,
            help="Reference size of fonts for all title/label on the figure")
    parser.add_argument("--baseline-type", default="auto",
            choices=["auto", "critic", "rollout", "nearnb", "none"],
            help="Baseline type (auto: read from args.json or detect from data)")
    return parser.parse_args()


def read_baseline_from_config(result_path):
    """Try to read baseline type from args.json in same directory"""
    output_dir = os.path.dirname(result_path)
    args_json_path = os.path.join(output_dir, 'args.json')
    
    if os.path.exists(args_json_path):
        try:
            with open(args_json_path) as f:
                config = json.load(f)
                if 'baseline_type' in config:
                    baseline_type = config['baseline_type']
                    print(f" Read baseline type from args.json: '{baseline_type}'")
                    return baseline_type
        except Exception as e:
            print(f"  Could not read args.json: {e}")
    
    return None


def detect_baseline_type(results):
    """Auto-detect baseline type from available columns"""
    
    # Check if BL column exists and has non-zero values
    if hasattr(results, 'bl'):
        bl_values = results.bl
        if np.any(np.abs(bl_values) > 1e-6):
            return "critic"  # Critic baseline (has learned values)
    
    # If BL exists but is all zeros/small, likely no baseline or simple baseline
    return "none"


def get_baseline_label(baseline_type):
    """Get appropriate label for baseline"""
    labels = {
        "critic": "estimated by critic on train data",
        "rollout": "estimated by rollout baseline on train data",
        "nearnb": "nearest neighbor baseline on train data",
        "hybrid": "hybrid baseline on train data",

        "none": None
    }
    return labels.get(baseline_type)


def main(args):
    mpl.rc('font', size=args.font_size)
    
    # Step 1: Try to read from args.json if auto mode
    if args.baseline_type == "auto":
        baseline_from_config = read_baseline_from_config(args.result_path)
        if baseline_from_config:
            baseline_type = baseline_from_config
        else:
            baseline_type = None  # Will detect from data
    else:
        baseline_type = args.baseline_type
        print(f" Using manually specified baseline type: '{baseline_type}'")
    
    # Read CSV
    with open(args.result_path) as f:
        header = next(f)
        Result = namedtuple("Result", [col.strip(" \n#").lower() for col in header.split() if col])
        results = []
        for row in f:
            results.append(Result(*(float(val) for val in row.split() if val)))
    results = Result(*(np.array(res) for res in zip(*results)))
    
    # Step 2: If still auto, detect from data
    if baseline_type is None or baseline_type == "auto":
        baseline_type = detect_baseline_type(results)
        print(f" Auto-detected baseline type from data: '{baseline_type}'")
    
    # Check which columns are available
    has_test = hasattr(results, 'test_mu')
    has_baseline = hasattr(results, 'bl') and get_baseline_label(baseline_type) is not None
    has_loss = hasattr(results, 'loss')
    has_prob = hasattr(results, 'prob')
    has_norm = hasattr(results, 'norm')
    
    print(f"\n Available data:")
    print(f"   Baseline: {baseline_type}")
    print(f"   Test data: {has_test}")
    print(f"   Baseline values: {has_baseline}")
    print(f"   Loss: {has_loss}")
    print(f"   Probability: {has_prob}")
    print(f"   Gradient norm: {has_norm}\n")
    
    # Create figure
    fig = plt.figure(constrained_layout=True)
    grid = fig.add_gridspec(nrows=3, ncols=4)

    # ========== Main plot ==========
    ax = fig.add_subplot(grid[:,:3])
    
    # Always plot training performance
    ax.plot(results.ep, results.val, label="observed on train data", 
            color='r', linewidth=2)
    
    # Plot baseline if available
    if has_baseline:
        baseline_label = get_baseline_label(baseline_type)
        ax.plot(results.ep, results.bl, label=baseline_label, 
                color='b', linewidth=2)
    
    # Plot test performance if available
    if has_test:
        ax.plot(results.ep, -results.test_mu, label="observed on test data", 
                color='g', linewidth=2)
        
        # Optionally add confidence interval
        if hasattr(results, 'test_std'):
            ax.fill_between(results.ep, 
                          -results.test_mu - results.test_std,
                          -results.test_mu + results.test_std, 
                          color='g', alpha=0.2)
    
    ax.legend(loc="upper right")
    ax.set_ylabel("Mean cumulated reward")
    ax.set_xlabel("Training epoch")
    ax.grid(True, alpha=0.3)

    # ========== Side plots ==========
    side_plots_added = 0
    
    # Routes probability
    if has_prob:
        ax = fig.add_subplot(grid[side_plots_added, 3])
        ax.plot(results.ep, results.prob, color='purple', linewidth=1.5)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Routes prob.")
        ax.grid(True, alpha=0.3)
        side_plots_added += 1
    
    # Loss (only meaningful for critic baseline)
    if has_loss and baseline_type == "critic":
        ax = fig.add_subplot(grid[side_plots_added, 3])
        ax.plot(results.ep, results.loss, color='orange', linewidth=1.5)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("AC loss")
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        side_plots_added += 1
    
    # Gradient norm
    if has_norm:
        ax = fig.add_subplot(grid[side_plots_added, 3])
        ax.plot(results.ep, results.norm, color='brown', linewidth=1.5)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Grad. norm")
        ax.set_xlabel("Train. epoch")
        ax.grid(True, alpha=0.3)
        side_plots_added += 1

    fig.set_size_inches(16, 9)

    # Save
    if args.output_path is None:
        args.output_path = args.result_path.replace(".csv", ".pdf")
    
    fig.savefig(args.output_path, bbox_inches='tight')
    print(f" Plot saved to: {args.output_path}")
    
    # Show summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Baseline type: {baseline_type}")
    print(f"Total epochs: {len(results.ep)}")
    print(f"Train reward: {results.val[0]:.2f} → {results.val[-1]:.2f}")
    if has_test:
        print(f"Test reward: {-results.test_mu[0]:.2f} → {-results.test_mu[-1]:.2f}")
    if has_baseline:
        print(f"Baseline converged: {abs(results.bl[-1] - results.val[-1]) < 1.0}")
    print("="*50 + "\n")
    
    plt.show()


if __name__ == "__main__":
    main(parse_args())