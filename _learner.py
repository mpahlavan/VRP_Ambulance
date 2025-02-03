from marpdan.layers import TransformerEncoder, MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLearner(nn.Module):
    def __init__(self, cust_feat_size, veh_state_size, model_size=128,
                 layer_count=3, head_count=8, ff_size=512, tanh_xplor=10, greedy=False):
        super().__init__()
        self.model_size = model_size
        self.inv_sqrt_d = model_size ** -0.5
        self.tanh_xplor = tanh_xplor
        
        # Enhanced customer embeddings
        self.depot_embedding = nn.Linear(cust_feat_size-1, model_size)  # Exclude spoilage time
        self.cust_embedding = nn.Linear(cust_feat_size-1, model_size)
        self.spoilage_encoder = nn.Sequential(
            nn.Linear(1, model_size//2),
            nn.ReLU(),
            nn.Linear(model_size//2, model_size)
        )
        
        # Customer encoder with temporal attention
        self.cust_encoder = TransformerEncoder(layer_count, head_count, model_size*2, ff_size)
        
        # Vehicle network with capacity awareness
        self.veh_state_encoder = nn.Linear(veh_state_size, model_size)
        self.fleet_attention = MultiHeadAttention(head_count, model_size, model_size)
        self.veh_attention = MultiHeadAttention(head_count, model_size)
        
        # Capacity-spoilage projection
        self.capacity_aware_proj = nn.Linear(model_size + 1, model_size)
        
        self.greedy = greedy
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _encode_customers(self, customers, mask=None):
        """Enhanced encoding with temporal urgency"""
        # Split features
        loc_features = customers[:, :, :3]  # x, y, demand
        spoilage_times = customers[:, :, 3:]  # spoilage time
        
        # Embed locations and spoilage separately
        depot_emb = self.depot_embedding(loc_features[:, 0:1])
        cust_emb = self.cust_embedding(loc_features[:, 1:])
        spoilage_emb = self.spoilage_encoder(spoilage_times)
        
        # Combine embeddings
        full_emb = torch.cat((
            depot_emb,
            cust_emb + spoilage_emb[:, 1:]  # Add spoilage to customer embeddings
        ), dim=1)
        
        # Temporal attention
        self.cust_enc = self.cust_encoder(
            torch.cat((full_emb, spoilage_emb), dim=-1),  # [batch, nodes, 2*model_size]
            mask
        )
        self.spoilage_bias = (1 - spoilage_times.squeeze(-1))  # Higher bias for urgent items

    def _repr_vehicle(self, vehicles, veh_idx, mask):
        """Capacity-aware vehicle representation"""
        # Encode vehicle state
        veh_emb = self.veh_state_encoder(vehicles)  # [batch, veh, model_size]
        
        # Add capacity information
        capacity_ratio = (vehicles[:, :, 2:3] / self.veh_capa)  # [batch, veh, 1]
        veh_emb = self.capacity_aware_proj(
            torch.cat((veh_emb, capacity_ratio), -1)
        )
        
        # Fleet attention
        fleet_repr = self.fleet_attention(
            veh_emb, veh_emb, veh_emb, 
            key_padding_mask=mask.all(-1)  # Mask fully blocked vehicles
        )
        
        # Current vehicle focus
        veh_query = fleet_repr.gather(1, veh_idx.unsqueeze(-1).expand(-1, -1, self.model_size))
        return self.veh_attention(veh_query, fleet_repr, fleet_repr)

    def _score_customers(self, veh_repr):
        """Urgency-prioritized scoring"""
        compat = veh_repr.matmul(self.cust_enc.transpose(1, 2)) * self.inv_sqrt_d
        
        # Add spoilage urgency bias
        compat += self.spoilage_bias.unsqueeze(1) * 2  # Scale urgency effect
        
        if self.tanh_xplor:
            compat = self.tanh_xplor * torch.tanh(compat)
            
        return compat

    def _get_logp(self, compat, veh_mask):
        compat = compat.masked_fill(veh_mask, -float('inf'))
        return F.log_softmax(compat.squeeze(1), dim=-1)

    def step(self, dyna):
        veh_repr = self._repr_vehicle(dyna.vehicles, dyna.cur_veh_idx, dyna.mask)
        compat = self._score_customers(veh_repr)
        logp = self._get_logp(compat, dyna.cur_veh_mask)
        
        if self.greedy:
            cust_idx = logp.argmax(-1, keepdim=True)
        else:
            cust_idx = torch.multinomial(logp.exp(), 1)
            
        return cust_idx, logp.gather(-1, cust_idx)

    def forward(self, dyna):
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