# critic.py
from marpdan.baselines._base import Baseline
import torch
import torch.nn as nn

class CriticBaseline(Baseline):
    def __init__(self, learner, cust_count, use_qval=True, use_cumul_reward=False):
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval
        # Match modified customer encoding dimensions
        self.project = nn.Linear(self.learner.model_size, cust_count+1 if use_qval else 1)

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        # Handle new vehicle state dimensions
        compat = learner_compat.clone()
        compat[vrp_dynamics.cur_veh_mask] = -float('inf')
        
        val = self.project(compat)
        if self.use_qval:
            val = val.gather(2, cust_idx.unsqueeze(1))
        return val.squeeze(1)

    def __call__(self, vrp_dynamics):
        # Align with new environment state tracking
        vrp_dynamics.nodes = vrp_dynamics.nodes.to(self.learner.device)
        if vrp_dynamics.cust_mask is not None:
            vrp_dynamics.cust_mask = vrp_dynamics.cust_mask.to(self.learner.device)
            
        return super().__call__(vrp_dynamics)

    def to(self, device):
        super().to(device)
        self.learner.to(device)