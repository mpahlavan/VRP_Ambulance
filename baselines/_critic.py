from marpdan.baselines._base import Baseline

import torch
import torch.nn as nn
import torch.nn.functional as F # Added for ReLU activation

class CriticBaseline(Baseline):
    def __init__(self, learner, cust_count, use_qval=True, use_cumul_reward=True, hidden_size=128, num_layers=2):
        """
        Initializes the CriticBaseline.

        Args:
            learner: The AttentionLearner (Actor) instance.
            cust_count: Number of customers (nodes - 1).
            use_qval: If True, the critic estimates Q-values (per action). If False, it estimates V-values (per state).
                      (Your config uses False, so it estimates V-values).
            use_cumul_reward: If True, uses cumulative rewards. If False, uses immediate rewards (handled by reinforce_loss).
            hidden_size: Dimension of the hidden layers in the Critic's MLP.
            num_layers: Number of hidden layers in the Critic's MLP.
        """
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval
        
       
        output_dim = cust_count + 1 if use_qval else 1
        
        
        input_dim = cust_count + 1 
        
        # Build the Multi-Layer Perceptron (MLP) for the critic
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU()) # Non-linear activation for hidden layers

        # Hidden layers
        for _ in range(num_layers - 1): # If num_layers is 1, this loop won't run.
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        # No activation function on the output layer for value estimation (regression task)
        layers.append(nn.Linear(hidden_size, output_dim)) 

        self.project = nn.Sequential(*layers)
        
        # Optional: Initialize weights for better training stability
        for m in self.project:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # Glorot uniform initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0) # Initialize biases to zero

    def eval_step(self, vrp_dynamics, learner_compat, cust_idx):
        compat = learner_compat.clone()
        compat[vrp_dynamics.cur_veh_mask] = 0

        val = self.project(compat)
        if self.use_qval:
            val = val.gather(2, cust_idx.unsqueeze(1).expand(-1,1,-1))
        return val.squeeze(1)

    def __call__(self, vrp_dynamics):
        mask = vrp_dynamics.mask if hasattr(vrp_dynamics, 'mask') else None
        self.learner._encode_customers(vrp_dynamics.nodes, mask )
        
        vrp_dynamics.reset()
        actions, logps, rewards, bl_vals = [], [], [], []
        
        while not vrp_dynamics.done:
            veh_repr = self.learner._repr_vehicle(
                    vrp_dynamics.vehicles,
                    vrp_dynamics.cur_veh_idx,
                    vrp_dynamics.mask)
            compat = self.learner._score_customers(veh_repr)
            logp = self.learner._get_logp(compat, vrp_dynamics.cur_veh_mask)
            #cust_idx = logp.exp().multinomial(1)
            
            # Safe probability calculation
            probs = logp.exp()  # Convert log probabilities to probabilities
            
            # Handle numerical issues
            probs[torch.isnan(probs) | torch.isinf(probs)] = 0.0
            probs[probs < 0] = 0.0
            
            # If no valid actions, force return to depot
            if probs.sum() < 1e-10:
                cust_idx = torch.zeros_like(vrp_dynamics.cur_veh_idx)
            else:
                # Normalize probabilities
                probs = probs / probs.sum(dim=1, keepdim=True)
                
                # Ensure depot is always an option with small probability
                probs[:, 0] = probs[:, 0].clone() + 1e-6
                probs = probs / probs.sum(dim=1, keepdim=True)
                
                try:
                    cust_idx = probs.multinomial(1)
                except RuntimeError:
                    # Fallback to depot if sampling fails
                    cust_idx = torch.zeros_like(vrp_dynamics.cur_veh_idx)

            if not(self.use_cumul and bl_vals):
                bl_vals.append( self.eval_step(vrp_dynamics, compat, cust_idx) )
            
            actions.append( (vrp_dynamics.cur_veh_idx, cust_idx) )
            logps.append( logp.gather(1, cust_idx) )
            rewards.append(vrp_dynamics.step(cust_idx))

        if self.use_cumul:
            rewards = torch.stack(rewards).sum(dim = 0)
            bl_vals = bl_vals[0]
        return actions, logps, rewards, bl_vals

    def parameters(self):
        return self.project.parameters()

    def state_dict(self):
        return self.project.state_dict()

    def load_state_dict(self, state_dict):
        return self.project.load_state_dict(state_dict)

    def to(self, device):
        self.project.to(device = device)
