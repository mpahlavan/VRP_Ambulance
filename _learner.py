from marpdan.layers import TransformerEncoder, MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLearner(nn.Module):
   def __init__(self, cust_feat_size, veh_state_size, model_size=128,
           layer_count=3, head_count=8, ff_size=512, tanh_xplor=10, greedy=False):
       """
       Attention-based learner for PVRP
       cust_feat_size: size of customer features (x,y, demand, spoilage)
       veh_state_size: size of vehicle state (x,y, time, capacity)
       """
       super().__init__()
       self.model_size = model_size
       self.inv_sqrt_d = model_size ** -0.5
       self.tanh_xplor = tanh_xplor
       
       # Embeddings
       self.depot_embedding = nn.Linear(cust_feat_size, model_size)
       self.cust_embedding = nn.Linear(cust_feat_size, model_size)
       
       # Customer encoder
       self.cust_encoder = TransformerEncoder(layer_count, head_count, model_size, ff_size)
       
       # Vehicle attention
       self.fleet_attention = MultiHeadAttention(head_count, veh_state_size, model_size)
       self.veh_attention = MultiHeadAttention(head_count, model_size)
       self.cust_project = nn.Linear(model_size, model_size)
       
       self.greedy = greedy
       self.init_parameters()

   def init_parameters(self):
       """Initialize network parameters"""
       for name, param in self.named_parameters():
           if param.dim() > 1:
               nn.init.xavier_uniform_(param)

   def _encode_customers(self, customers, mask=None):
       """
       Encode customer features
       customers: [batch, nodes, features] including depot at index 0
       mask: [batch, nodes] binary mask for dynamic/hidden customers
       """
       # Embed depot and customers separately
       cust_emb = torch.cat((
           self.depot_embedding(customers[:, 0:1, :]),  # [batch, 1, model_size]
           self.cust_embedding(customers[:, 1:, :])     # [batch, nodes-1, model_size]
       ), dim=1)  # [batch, nodes, model_size]
       
       # Apply mask if provided
       if mask is not None:
           cust_emb = cust_emb.masked_fill(mask.unsqueeze(-1), 0)
           
       # Encode using transformer
       self.cust_enc = self.cust_encoder(cust_emb, mask)  # [batch, nodes, model_size]
       
       # Precompute fleet attention
       self.fleet_attention.precompute(self.cust_enc)
       
       # Project customer representations
       self.cust_repr = self.cust_project(self.cust_enc)  # [batch, nodes, model_size]
       if mask is not None:
           self.cust_repr = self.cust_repr.masked_fill(mask.unsqueeze(-1), 0)

   def _repr_vehicle(self, vehicles, veh_idx, mask):
       """
       Get vehicle representation
       vehicles: [batch, veh_count, state_size]
       veh_idx: [batch, 1]
       mask: [batch, veh_count, nodes]
       """
       # Get fleet representation with attention
       fleet_repr = self.fleet_attention(vehicles, mask=mask)  # [batch, veh_count, model_size]
       
       # Get current vehicle query
       veh_query = fleet_repr.gather(
           1, 
           veh_idx.unsqueeze(2).expand(-1, -1, self.model_size)
       )  # [batch, 1, model_size]
       
       # Attend to fleet
       return self.veh_attention(
           veh_query, 
           fleet_repr, 
           fleet_repr
       )  # [batch, 1, model_size]

   def _score_customers(self, veh_repr):
       """
       Score customers for selection
       veh_repr: [batch, 1, model_size]
       Returns: [batch, 1, nodes]
       """
       # Calculate compatibility scores
       compat = veh_repr.matmul(self.cust_repr.transpose(1, 2))  # [batch, 1, nodes]
       compat *= self.inv_sqrt_d
       
       # Apply tanh exploration if enabled
       if self.tanh_xplor is not None:
           compat = self.tanh_xplor * compat.tanh()
           
       return compat

   def _get_logp(self, compat, veh_mask):
       
        """
        Get log probabilities for customer selection
        compat: [batch, 1, nodes]
        veh_mask: [batch, 1, nodes]
        Returns: [batch, nodes]
        """
        # Mask infeasible customers
        compat = compat.masked_fill(veh_mask, -float('inf'))
        

        #print("Compat shape:", compat.shape)
        #print("Vehicle mask shape:", veh_mask.shape)

        # Check if mask is all False
        #print("Mask any True:", veh_mask.any())

        # Get log probabilities
        return F.log_softmax(compat, dim=2).squeeze(1)

   def step(self, dyna):
       """
       Execute single step
       Returns: customer index and log probability
       """
       # Get vehicle representation
       veh_repr = self._repr_vehicle(
           dyna.vehicles, 
           dyna.cur_veh_idx,
           dyna.mask
       )
       
       # Score customers
       compat = self._score_customers(veh_repr)
       
       # Get action probabilities
       logp = self._get_logp(compat, dyna.cur_veh_mask)
       
       # Select action
       if self.greedy:
           cust_idx = logp.argmax(dim=1, keepdim=True)
       else:
           cust_idx = logp.exp().multinomial(1)
           
       return cust_idx, logp.gather(1, cust_idx)

   def forward(self, dyna):
       """
       Full episode rollout
       Returns: actions, log probabilities, rewards
       """
       dyna.reset()
       actions, logps, rewards = [], [], []
       
       while not dyna.done:
            if dyna.new_customers:
               self._encode_customers(dyna.nodes, dyna.cust_mask)
               
            cust_idx, logp = self.step(dyna)
            actions.append((dyna.cur_veh_idx, cust_idx))
            logps.append(logp)
            rewards.append(dyna.step(cust_idx))
           
       return actions, logps, rewards