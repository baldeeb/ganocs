    
import torch

from .vanilla import Discriminator
from .multi import MultiDiscriminatorWithOptimizer


# NOTE: temp hacky solution. this is not the way to include depth
class RgbdMultiDiscriminatorWithOptimizer(MultiDiscriminatorWithOptimizer):
    def __init__(self, 
                 in_ch=3, 
                 feat_ch=64,
                 num_classes=10,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 logger=None
                ):
        discs = [Discriminator(in_ch+1, feat_ch) for _ in range(num_classes)]
        super().__init__(discs, num_classes, optimizer, optim_args, logger)

    @property
    def properties(self):
        return super().properties + ['depth_context']