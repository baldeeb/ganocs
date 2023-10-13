import torch
from torch import nn

class DepthConvHead(nn.Module):
    def __init__(self, ch_in, ch_hidden, ch_out=1):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        activation_layer = nn.SELU()
        _conv = lambda i, o: nn.Sequential(
                                nn.Conv2d( i, o, 3, 1, 1),
                                norm_layer(o), activation_layer
                            )
        _down_conv = lambda i, o : nn.Sequential(
                                        nn.Conv2d(i, o, 3, 2, 1),
                                        norm_layer(o), activation_layer,
                                    )

        self.layers = nn.Sequential(
            _down_conv(ch_in,       ch_hidden),
            _down_conv(ch_hidden,   ch_hidden*2),
            _down_conv(ch_hidden*2, ch_hidden*2),
            _down_conv(ch_hidden*2, ch_hidden*2),
            nn.Flatten(1),
            nn.Linear(2*ch_hidden*4, ch_out),
        )
    
    def forward(self, x):
        return self.layers(x)