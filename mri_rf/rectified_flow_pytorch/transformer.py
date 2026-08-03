import torch
from torch import nn

class Transformer_2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, flow):
        return x

class Transformer_3D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, flow):
        return x

class Transformer_2D_win(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, flow):
        return x
