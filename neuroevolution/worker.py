# neuroevolution/worker.py

import torch
import torch.nn as nn
from torch.optim import Adam
from copy import deepcopy

class PBTWorker:
    """
   Population-Based Training
    
     worker cosist of:
    - Policy (AttentionLearner)
    - Baseline (Critic)
    - Optimizer
    - Hyperparameters
    - Performance tracking
    """
    
    def __init__(self, worker_id, learner, baseline, hyperparams, device):
        """
        Args:
            worker_id:  worker (0, 1, 2, ...)
            learner: AttentionLearner instance
            baseline: Baseline instance (critic)
            hyperparams: dict with hyperparameters
            device: cuda  cpu
        """
        print(f"\n  ===== PBTWorker.__init__ for worker {worker_id} =====")
        print(f"    Received baseline: {baseline}")
        print(f"    baseline type: {type(baseline)}")
        print(f"    baseline is None: {baseline is None}")
        
        if baseline is None:
            raise ValueError(f" Baseline passed to PBTWorker is None!")
        
        self.worker_id = worker_id
        self.device = device
        
        # Policy و Baseline
        print(f"    About to deepcopy learner...")
        try:
            self.learner = deepcopy(learner).to(device)
            print(f"     Learner deepcopied successfully")
        except Exception as e:
            print(f"     Error deepcopying learner: {e}")
            raise
        
        print(f"    About to deepcopy baseline...")
        print(f"    baseline before deepcopy: {baseline}")
        print(f"    baseline is None before deepcopy: {baseline is None}")
        
        try:
            print(f"    Calling deepcopy(baseline)...")
            baseline_copy = deepcopy(baseline)
            print(f"     Deepcopy returned: {baseline_copy}")
            print(f"    baseline_copy is None: {baseline_copy is None}")
            print(f"    baseline_copy type: {type(baseline_copy)}")
            
            print(f"    Calling .to(device)...")
            self.baseline = baseline_copy.to(device)
            print(f"     .to(device) successful")
            print(f"    self.baseline: {self.baseline}")
            print(f"    self.baseline is None: {self.baseline is None}")
            
        except Exception as e:
            print(f"    Error during deepcopy/to: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if self.baseline is None:
            raise ValueError(f" self.baseline is None after deepcopy!")
        
        # Hyperparameters (mutable)
        self.hyperparams = hyperparams.copy()
        
        # Optimizer با LR از hyperparams
        print(f"    Creating optimizer...")
        try:
            print(f"    Getting learner parameters...")
            learner_params = list(self.learner.parameters())
            print(f"     Learner has {len(learner_params)} parameters")
            
            print(f"    Getting baseline parameters...")
            print(f"    self.baseline: {self.baseline}")
            print(f"    self.baseline is None: {self.baseline is None}")
            print(f"    hasattr parameters: {hasattr(self.baseline, 'parameters')}")
            
            baseline_params = list(self.baseline.parameters())
            print(f"     Baseline has {len(baseline_params)} parameters")
            
            self.optimizer = Adam([
                {"params": self.learner.parameters(), 
                "lr": self.hyperparams['actor_lr']},
                {"params": self.baseline.parameters(), 
                "lr": self.hyperparams['critic_lr']}
            ])
            print(f"    Optimizer created")
            
        except Exception as e:
            print(f"     Error creating optimizer: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Performance tracking (lower cost is better, so initialize to +inf)
        self.performance_history = []
        self.best_performance = float('inf')  # Changed from -inf to +inf for cost minimization
        self.steps = 0
        
        print(f"   PBTWorker {worker_id} initialized successfully\n")
            
    def get_performance(self):
        """Get current performance (lower is better for cost)"""
        if not self.performance_history:
            return float('inf')  # Changed from -inf to +inf for cost minimization
        return self.performance_history[-1]

    def update_performance(self, perf):
        """Update performance (lower is better for cost)"""
        self.performance_history.append(perf)
        if perf < self.best_performance:  # Changed from > to < for cost minimization
            self.best_performance = perf
        self.steps += 1
    
    def update_hyperparams(self, new_hyperparams):
        self.hyperparams = new_hyperparams.copy()
        
        # Rebuild optimizer
        self.optimizer = Adam([
            {"params": self.learner.parameters(), 
             "lr": self.hyperparams['actor_lr']},
            {"params": self.baseline.parameters(), 
             "lr": self.hyperparams['critic_lr']}
        ])
    
    def copy_from(self, other_worker):
        """کپی کردن policy و baseline از worker دیگر"""
        self.learner.load_state_dict(other_worker.learner.state_dict())
        self.baseline.load_state_dict(other_worker.baseline.state_dict())
    
    def state_dict(self):
        return {
            'learner': self.learner.state_dict(),
            'baseline': self.baseline.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'hyperparams': self.hyperparams,
            'performance_history': self.performance_history,
            'best_performance': self.best_performance,
            'steps': self.steps
        }
    
    def load_state_dict(self, state_dict):
        self.learner.load_state_dict(state_dict['learner'])
        self.baseline.load_state_dict(state_dict['baseline'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.hyperparams = state_dict['hyperparams']
        self.performance_history = state_dict['performance_history']
        self.best_performance = state_dict['best_performance']
        self.steps = state_dict['steps']