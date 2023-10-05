from torchvision.ops import misc as misc_nn_ops
from typing import Optional, Callable, Dict
from torch import nn
import torch 

class NocsHeads(nn.Module):
    '''Derived from torchvision.models.detection.mask_rcnn.(MaskRCNNHeads, MaskRCNNPredictor)'''
    def __init__(self, 
                 in_channels, 
                 layers=(256, 256, 256, 256, 256), 
                 num_classes=91,
                 num_bins=32,
                 keys=['x', 'y', 'z'], 
                 multiheaded=True,
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
        else: last_activation = None

        if multiheaded: 
            self.head = self._get_multi_head(keys, 
                                             in_channels, 
                                             layers, 
                                             self.num_classes * self.num_bins,
                                             dilation=1, 
                                             norm_layer=norm_layer,
                                             last_activation=last_activation,
                                            )
        else:
            self.head = self._get_head(in_channels,
                                       layers, 
                                       len(keys) * self.num_bins * self.num_classes,
                                       dilation=1,
                                       norm_layer=norm_layer,
                                       last_activation=last_activation,
                                      )

            
        self._normal_init(self.modules())

    def __getitem__(self, key): return self.head[key]

    def _get_multi_head(self, keys, in_ch, layers, out_ch, dilation, norm_layer, last_activation=None):
        return nn.ModuleDict(
            {k: self._get_head(in_ch, layers, out_ch, dilation, norm_layer,
                               last_activation=last_activation)
            for k in keys})        

    def _get_head(self, in_ch, layers, out_ch, dilation, norm_layer, last_activation):
        # m = self._original(in_ch, layers, out_ch, dilation, norm_layer)
        m = self._experimental(in_ch, layers, out_ch, dilation, norm_layer)
            
        if last_activation is not None:
            m.add_module('last_activation', last_activation)
        return m
    
    def _original(self, in_ch, layers, out_ch, dilation, norm_layer):
        def _head_piece(in_ch, layers, dilation, norm_layer):
            blocks = []
            next_feature = in_ch
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
        def _predictor(in_ch, feat_ch, out_ch):
            blocks = [
                nn.ConvTranspose2d(in_ch, feat_ch, 2, 2, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(feat_ch, out_ch, 1, 1, 0),
            ]
            return nn.Sequential(*blocks)
        
        h = _head_piece(in_ch, layers[:-1], dilation=dilation, 
                            norm_layer=norm_layer)
        p = _predictor(layers[-2], layers[-1], out_ch)
        return nn.Sequential(h, p) 

    def _experimental(self, in_ch, layers, out_ch, dilation, norm_layer):
        activation_layer = nn.LeakyReLU(0.2)
        _conv = lambda i, o: nn.Sequential(
                                nn.Conv2d( i, o, kernel_size=3, stride=1,
                                    padding=dilation, dilation=dilation,),
                                norm_layer(o), activation_layer
                            )
        _down_conv = lambda i, o : nn.Sequential(
                                        nn.Conv2d(i, o, 2, 2, 0),
                                        norm_layer(o), activation_layer,
                                    )
        _trans_conv = lambda i, o : nn.Sequential(
                                        nn.ConvTranspose2d(i, o, 2, 2, 0),
                                        norm_layer(o), activation_layer,
                                    )
        _last_conv = lambda i: nn.Sequential(
                                    nn.Conv2d( i, i, kernel_size=3, stride=1,
                                            padding=dilation, dilation=dilation,),
                                    activation_layer,
                                    nn.Conv2d(i, out_ch, kernel_size=1, stride=1,
                                        padding=0, dilation=dilation,)
                                )
        blocks = [
            _conv(in_ch, 256),
            _conv(256,   256),
            _conv(256,   256),
            _conv(256,   512),
            # _down_conv(512, 1028),
            _trans_conv(512, 256),
            # _trans_conv(512, 256),
            _conv(256, 256),
            _conv(256, 128),
            _last_conv(128),
        ]

        return nn.Sequential(*blocks)

    def _init_weights(self, modules, fx):
         for m in modules:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight = fx(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _kaiming_normal_init(self, modules):
        init_fx = lambda w : nn.init.kaiming_normal_(w, 
                                                     mode="fan_out", 
                                                     nonlinearity="relu")
        self._init_weights(modules, init_fx)

    def _normal_init(self, modules):
        init_fx = lambda w : nn.init.normal_(w, std=0.02)
        self._init_weights(modules, init_fx)
        

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