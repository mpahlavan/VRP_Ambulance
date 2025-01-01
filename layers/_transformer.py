from marpdan.layers import MultiHeadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, head_count, model_size, ff_size, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(head_count, model_size)
        self.bn1 = nn.BatchNorm1d(model_size)

        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.ff1 = nn.Linear(model_size, ff_size)
        self.ff2 = nn.Linear(ff_size, model_size)
        self.bn2 = nn.BatchNorm1d(model_size)

        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, h_in, mask = None):
        r"""
        :param proj_in: :math:`N \times L \times D_M`
        :param mask:    :math:`N \times L`
        :return:        :math:`N \times L \times D_M`
        """
        att = self.mha(h_in, mask = mask)
        att = self.dropout1(att)
        att = self.bn1( (h_in + att).permute(0,2,1) ).permute(0,2,1)

        h_out = F.relu( self.ff1(att))
        h_out = self.dropout2(h_out)  
        h_out = self.ff2(h_out)
        h_out = self.bn2( (att + h_out).permute(0,2,1) ).permute(0,2,1)

        if mask is not None:
            h_out[mask] = 0
        return h_out


class TransformerEncoder(nn.Module):
    r"""Neural Network module implementing a self-attention mechanism used as encoder.
    This layer structure was first introduced in "Attention Is All You Need" by \
            `[Vaswani et al. (2017)] <http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`_
    """
    def __init__(self, layer_count, head_count, model_size, ff_size, dropout_rate=0.1, max_grad_norm=1.0):
        super().__init__()
        self.max_grad_norm = max_grad_norm

        for l in range(layer_count):
            self.add_module(
                str(l), 
                TransformerEncoderLayer(head_count, model_size, ff_size, dropout_rate))

    def forward(self, inputs, mask = None):
        r"""
        :param inputs: :math:`N \times L \times D_M`
        :param mask:   :math:`N \times L`
        :return:       :math:`N \times L \times D_M`
        """
        h = inputs
        for child in self.children():
            h = child(h, mask)
        
        if self.training:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=self.max_grad_norm
            )
            
        return h
