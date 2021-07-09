import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class InnerProductDecoder(nn.Module):
    """using inner product for predicting structures."""

    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.fudge = 1e-7

    def forward(self, z):
 
        adj = torch.sigmoid(torch.matmul(z,z.t()))
        return adj
