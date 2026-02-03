#!/usr/bin/env python3

import sys
sys.path.insert(0, '/py_ws/marpdan')

from marpdan import AttentionLearner
from marpdan.baselines._critic import CriticBaseline
import torch

print("="*60)
print("Testing CriticBaseline Creation")
print("="*60)

# Create learner
print("\n1. Creating learner...")
try:
    learner = AttentionLearner(
        cust_feat_size=4,
        veh_state_size=4,
        model_size=128,
        layer_count=3,
        head_count=8,
        ff_size=512,
        tanh_xplor=11
    )
    print(f"    Learner created: {type(learner)}")
except Exception as e:
    print(f"    Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create baseline
print("\n2. Creating baseline...")
print(f"   Parameters:")
print(f"     learner: {learner}")
print(f"     cust_count: 10")
print(f"     use_qval: False")
print(f"     use_cumul_reward: True")
print(f"     hidden_size: 128")
print(f"     num_layers: 2")

try:
    baseline = CriticBaseline(
        learner,
        10,  # cust_count
        use_qval=False,
        use_cumul_reward=True,
        hidden_size=128,
        num_layers=2
    )
    print(f"    Baseline created successfully!")
    print(f"   Type: {type(baseline)}")
    print(f"   baseline object: {baseline}")
    print(f"   baseline is None: {baseline is None}")
    
    # Test parameters
    if baseline is not None:
        try:
            params = list(baseline.parameters())
            print(f"    Baseline has {len(params)} parameter tensors")
            print(f"   First param shape: {params[0].shape if params else 'N/A'}")
        except Exception as e:
            print(f"    Error getting parameters: {e}")
    else:
        print(f"    WARNING: Baseline is None!")
    
except Exception as e:
    print(f"    Error creating baseline: {e}")
    import traceback
    traceback.print_exc()
    baseline = None

# Test deepcopy
if baseline is not None:
    print("\n3. Testing deepcopy...")
    try:
        from copy import deepcopy
        baseline_copy = deepcopy(baseline)
        print(f"    Deepcopy successful!")
        print(f"   Type: {type(baseline_copy)}")
        
        # Test to device
        device = torch.device('cpu')
        baseline_gpu = baseline_copy.to(device)
        print(f"    .to(device) successful!")
        
    except Exception as e:
        print(f"    Error in deepcopy: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Test Complete")
print("="*60)
