# import torch
from torch import nn

from .film import FiLMLayer
# from .multi import MultiDiscriminatorWithOptimizer
# from .optimizer_wrapper import DiscriminatorWithOptimizer
from .depth_encoder import DepthConvHead

class ContextAwareDiscriminator(nn.Module):
    '''Discriminator built for FiLM layers. Accepts depth as context.'''
    def __init__(self, in_ch, feat_ch, dim_ctx, out_ch=1):
        super().__init__()
        def _f(i, f, k=4, s=2, p=1, act=True, bn=True):  
            '''Helper for layer definition'''
            return FiLMLayer(in_ch=i, 
                      out_ch=f,
                      kernel_size=k, 
                      stride=s,
                      padding=p, 
                      dim_ctx=dim_ctx,
                      activation=nn.LeakyReLU(0.2, inplace=True) \
                                 if act else None,
                      batch_norm=nn.BatchNorm2d if bn else None
            )
        self._contextual_layers = nn.ModuleList([
            _f(in_ch,     feat_ch,                  bn=False), # no batchnorm
            _f(feat_ch,   2*feat_ch,                bn=False),
            _f(2*feat_ch, 4*feat_ch,                bn=False),
            _f(4*feat_ch, out_ch,    k=3, s=1, p=0, bn=False), # final layer
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x, ctx):
        for m in self._contextual_layers:
            x = m(x, ctx)
        x = nn.functional.sigmoid(x)
        return x
    
class DepthAwareDiscriminator(ContextAwareDiscriminator):
    def __init__(self, in_ch=3, feat_ch=64, dim_ctx=64, out_ch=1):
        super().__init__(in_ch, feat_ch, dim_ctx, out_ch)
        self._depth_encoder = DepthConvHead(1, feat_ch, dim_ctx)
        self._init_weights()
    def forward(self, x, ctx):
        ctx = self._depth_encoder(ctx)
        return super().forward(x, ctx)
    
    @property
    def properties(self): return ['depth_context']

# class ContextualMultiDiscriminatorWithOptimizer(nn.Module):
#     def __init__(self, 
#                  discriminators,
#                  num_classes=10,
#                  optimizer=torch.optim.Adam,
#                  optim_args={'lr':1e-4, 
#                              'betas':(0.5, 0.999)},
#                  logger=None
#                 ):
#         super().__init__()
#         '''Assumes class zero is discarded/background as does MRCNN.'''
#         self.discriminators = nn.ModuleDict({
#             str(i): DiscriminatorWithOptimizer(
#                         discriminators[i], 
#                         optimizer, 
#                         optim_args, 
#                         logger
#                     )
#             for i in range(1, num_classes)
#         })
#         self.logger = logger
        
#     def _multihead_step(self, latents, is_real:bool, classes, **kwargs):
#         losses, values = [], []
#         for i, (c, z) in enumerate(zip(classes, latents)):
#             ith_kwargs = {k:v[i:i+1] for k,v in kwargs.items()}
#             disc = self.discriminators[str(c.item())]
#             loss, val = disc._step(z[None], is_real, ith_kwargs)
#             losses.append(loss.item()); values.append(val)
#         return torch.tensor(losses), torch.tensor(values)

#     def _log(self, *args, **kwargs):
#         return DiscriminatorWithOptimizer._log(self, *args, **kwargs)            

#     def _update(self, data, real, classes):
#         loss, val = self._multihead_step(data, real, classes)
#         self._log(loss, val, real)
#         return loss.mean().detach()

#     def forward(self, x, classes, **kwargs):
#         '''
#         args:
#             x: (B, C, H, W) tensor of images
#             classes: (B,) tensor of classes
#             kwargs: every argument is expected to be a list of the same length as classes.
#         returns:
#             (B,) tensor of losses
#         '''
#         losses = []
#         for i, (c, p) in enumerate(zip(classes, x)):
#             ith_kwargs = {k:v[i:i+1] for k,v in kwargs.items()}
#             losses.append(self.discriminators[str(c.item())].forward(p[None], **ith_kwargs))
#         return torch.cat(losses) if len(losses) > 0 else torch.tensor([])

