from marpdan.layers import TransformerEncoder, MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLearner(nn.Module):
    def __init__(self, cust_feat_size, veh_state_size, model_size=128,
                 layer_count=3, head_count=8, ff_size=512, tanh_xplor=11, greedy=False):
        super().__init__()
        self.model_size = model_size
        self.inv_sqrt_d = model_size ** -0.5
        self.tanh_xplor = tanh_xplor
        self.greedy = greedy

        # Embedding layers
        self.depot_embedding = nn.Linear(cust_feat_size, model_size)
        self.cust_embedding = nn.Linear(cust_feat_size, model_size)

        # Transformer encoder for customers
        self.cust_encoder = TransformerEncoder(layer_count, head_count, model_size, ff_size)

        # Attention layers
        self.fleet_attention = MultiHeadAttention(head_count, veh_state_size, model_size)
        self.veh_attention = MultiHeadAttention(head_count, model_size)
        self.cust_project = nn.Linear(model_size, model_size)

    def _encode_customers(self, customers, mask=None):
        """
        customers: [batch, nodes, cust_feat_size]
        mask: [batch, nodes]
        """
        # Embed depot and other customers
        cust_emb = torch.cat((
            self.depot_embedding(customers[:, 0:1, :]),
            self.cust_embedding(customers[:, 1:, :])
        ), dim=1)  # [batch, nodes, model_size]

        if mask is not None:
            cust_emb = cust_emb.masked_fill(mask.unsqueeze(-1), 0)

        self.cust_enc = self.cust_encoder(cust_emb, mask)  # [batch, nodes, model_size]
        self.fleet_attention.precompute(self.cust_enc)

        self.cust_repr = self.cust_project(self.cust_enc)  # [batch, nodes, model_size]

        if mask is not None:
            self.cust_repr = self.cust_repr.masked_fill(mask.unsqueeze(-1), 0)

    def _repr_vehicle(self, vehicles, veh_idx, mask):
        """
        vehicles: [batch, veh_count, veh_state_size]
        veh_idx: [batch, 1]
        mask: [batch, veh_count, nodes]
        """
        fleet_repr = self.fleet_attention(vehicles, mask=mask)  # [batch, veh_count, model_size]
        veh_query = fleet_repr.gather(1, veh_idx.unsqueeze(-1).expand(-1, -1, self.model_size))  # [batch, 1, model_size]
        return self.veh_attention(veh_query, fleet_repr, fleet_repr)  # [batch, 1, model_size]

    def _score_customers(self, veh_repr):
        """
        veh_repr: [batch, 1, model_size]
        Returns: [batch, 1, nodes]
        """
        compat = veh_repr.matmul(self.cust_repr.transpose(1, 2))  # [batch, 1, nodes]
        compat *= self.inv_sqrt_d

        if self.tanh_xplor is not None:
            compat = self.tanh_xplor * compat.tanh()

        return compat

    def _get_logp(self, compat, veh_mask):
        """
        compat: [batch, 1, nodes]
        veh_mask: [batch, 1, nodes]
        Returns: [batch, nodes]
        """
        compat = compat.masked_fill(veh_mask, -float('inf'))
        return compat.log_softmax(dim=2).squeeze(1)

    def step(self, dyna):
        veh_repr = self._repr_vehicle(dyna.vehicles, dyna.cur_veh_idx, dyna.mask)
        compat = self._score_customers(veh_repr)
        logp = self._get_logp(compat, dyna.cur_veh_mask)
        if self.greedy:
            cust_idx = logp.argmax(dim=1, keepdim=True)
        else:
            cust_idx = logp.exp().multinomial(1)
        return cust_idx, logp.gather(1, cust_idx)

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
