from torchvision.ops import misc as misc_nn_ops
from typing import Optional, Callable, Dict
from torch import nn
import torch 

class NocsHeads(nn.Module):
    '''Derived from torchvision.models.detection.mask_rcnn.(MaskRCNNHeads, MaskRCNNPredictor)'''
    def __init__(self, 
                 in_channels, 
                 layers, 
                 num_classes,
                 num_bins,
                 keys=['x', 'y', 'z'], 
                 multiheaded=False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
                 mode: str = 'classification',  # 'classification' or 'regression'
                 ):
        super().__init__()
        self.in_channels    = in_channels
        self.keys           = keys
        self.num_classes    = num_classes
        self.num_bins       = num_bins
        self.multiheaded    = multiheaded
        self.mode           = mode 

        if mode == 'regression':
            last_activation = nn.Tanh()
            assert self.num_bins == 1, 'Regression only supports 1 bin'

        if multiheaded: 
            self.head = self._get_multi_head(keys, in_channels, 
                                                layers, 
                                                self.num_classes * self.num_bins,
                                                dilation=1, 
                                                norm_layer=norm_layer,
                                                  last_activation=last_activation,
                                                  )
        else:
            self.head = self._get_single_head(keys, 
                                              in_channels,
                                              layers, 
                                              norm_layer=norm_layer,
                                              last_activation=last_activation,
                                            )

            
        self._normal_init(self.modules())

    def __getitem__(self, key): return self.head[key]

    def _get_single_head(self, keys, in_ch, layers, norm_layer, last_activation=None):
        nc = len(keys) * self.num_bins * self.num_classes
        return self._get_multi_head(['k'], 
                                    in_ch, 
                                    layers, 
                                    nc, 
                                    dilation=1, 
                                    norm_layer=norm_layer,
                                    last_activation=last_activation,
                                )['k']

    def _get_multi_head(self, keys, in_ch, layers, out_ch, dilation, norm_layer, last_activation=None):
        net = nn.ModuleDict({
            k: nn.Sequential(
                self._head_piece(in_ch, layers[:-1], 
                                 dilation=dilation, 
                                 norm_layer=norm_layer),
                self._predictor(layers[-2], 
                                layers[-1], 
                                out_ch),
            )
            for k in keys
        })
        if last_activation is not None:
            for k in keys:
                net[k].add_module('last_activation', last_activation)
        return net

    def _head_piece(self, in_channels, layers, dilation, norm_layer):
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
    
    def _predictor(self, in_ch, feat_ch, out_ch):
        blocks = [
            nn.ConvTranspose2d(in_ch, feat_ch, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, out_ch, 1, 1, 0),
        ]
        return nn.Sequential(*blocks)

    def _kaiming_normal_init(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)

    def _normal_init(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)

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
        results: Dict[str, torch.Tensor] = {}
        B = x.shape[0]
        if self.multiheaded:
            for k in self.keys:
                kv = self.head[k](x)
                results[k] = kv.reshape(B, self.num_classes, 
                                        self.num_bins, *kv.shape[-2:])
        else:
            kv = self.head(x) # [B, (keys x classes x bins), 28, 28]
            for i, k in enumerate(self.keys):
                kvr = kv.reshape(B, len(self.keys), self.num_classes, 
                                 self.num_bins, *kv.shape[-2:])
                results[k] = kvr[:, i, ...]  # [B, classes, bins, 28, 28]
                
        if self.mode == 'regression':
            results = {k: self._regression_postprocess(v) for k, v in results.items()}
        # elif self.mode == 'classification':
        #     results = {k: self._classification_postprocess(v) for k, v in results.items()}
        return results
        
    def _regression_postprocess(self, x):
        return ( x + 1.0 ) / 2.0
    
    def _classification_postprocess(self, x):
        return torch.softmax(x, dim=2)