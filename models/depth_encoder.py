import torch
from torch import nn

class DepthConvHead(nn.Module):
    def __init__(self, ch_in, ch_hidden, ch_out):
        self.activation = nn.SELU(),
        self.layers = nn.Sequential(
            nn.Conv2d(ch_in, ch_hidden, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch_hidden),
            nn.SELU(),
            nn.Conv2d(ch_hidden, ch_hidden, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch_hidden),
            nn.SELU(),
            nn.Conv2d(ch_hidden, ch_hidden, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch_hidden),
            nn.SELU(),
            nn.Conv2d(ch_hidden, ch_out, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
    
    def forward(self, x):
        x = self.layers(x)