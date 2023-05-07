from torchvision.ops import misc as misc_nn_ops
from typing import Optional, Callable, Dict
from torch import nn
import torch 

class NocsDetectionHead(nn.Module):
    '''Derived from torchvision.models.detection.mask_rcnn.(MaskRCNNHeads, MaskRCNNPredictor)'''
    def __init__(self, 
                 in_channels, 
                 layers, 
                 num_classes,
                 num_bins,
                 keys=['x', 'y', 'z'], 
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.keys = keys
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.parts = nn.ModuleDict({
            k: nn.Sequential(
                self._head(in_channels, layers[:-1], dilation=1, norm_layer=norm_layer),
                self._predictor(layers[-2], layers[-1], num_classes*num_bins),
            )
            for k in keys
        })

    def __getitem__(self, key):
        return self.parts[key]

    def _head(self, in_channels, layers, dilation, norm_layer):
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.append(
                misc_nn_ops.Conv2dNormActivation(
                    next_feature, layer_features,
                    kernel_size=3,stride=1,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
            next_feature = layer_features
        return nn.Sequential(*blocks)
    
    def _predictor(self, in_ch, feat_ch, num_classes):
        blocks = [
            nn.ConvTranspose2d(in_ch, feat_ch,
                kernel_size=2, stride=1, padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d( feat_ch, num_classes,
                kernel_size=1, stride=1, padding=0,
            )    
        ]
        return nn.Sequential(*blocks)

    def _kaiming_normal_init(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: 
                    nn.init.zero_(m.bias)

    def _normal_init(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: 
                    nn.init.zero_(m.bias)

    def forward(self, x):
        ''' Returns nocs features given mask features.
        Args:
            features (torch.Tensor) of shape [B, C, H, W]
                representing the region of interest.
        Returns:
            nocs_results (Dict[str, torch.Tensor]): each element
                of the dict has shape [B, C, N, H, W], where C is
                the number of classes predicted and N is the number
                of bins used for this binary nocs predictor. 
        '''
        B, _, H, W = x.shape
        results: Dict[str, torch.Tensor] = {}
        for k in self.keys:
            kv = self.parts[k](x)
            results[k] = kv.reshape(B, self.num_classes, 
                                    self.num_bins, H, W)
        return results