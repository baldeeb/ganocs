import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from models.film import FiLMLayer
import numpy as np


class Discriminator(nn.Module):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, in_ch, feat_ch=128, out_ch=1):
        super().__init__()

        # a 4x4 kernel model. closer to DCGAN.
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, 2, 1, bias=False),  # 28 -> 14
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),

            # NOTE: removed this layer to reduce params when using multiple
            # nn.Conv2d(feat_ch, feat_ch, 3, 1, 1, bias=False), # 14 -> 14
            # nn.BatchNorm2d(feat_ch),
            # nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv2d(feat_ch, feat_ch, 3, 1, 1, bias=False), # 14 -> 14
            # nn.BatchNorm2d(feat_ch),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_ch,   2*feat_ch, 3, 2, 1, bias=False), # 14 -> 7
            nn.BatchNorm2d(2*feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(2*feat_ch, 4*feat_ch, 3, 2, 1, bias=False), # 7 -> 4
            nn.BatchNorm2d(4*feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(4*feat_ch, 8*feat_ch, 3, 2, 1, bias=False), # 4 -> 2
            nn.BatchNorm2d(8*feat_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(1),
            nn.Linear(8*feat_ch*4, out_ch),

            # nn.Conv2d(8*feat_ch, 8*feat_ch, 2, 1, 0, bias=False), # 2 -> 1
            # nn.BatchNorm2d(8*feat_ch),
            # nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv2d(8*feat_ch, out_ch, 1, 1, 0, bias=False), # 1 -> 1
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def _accuracy(value:torch.Tensor, real:bool):
        value = value.flatten()
        if real:
            acc = torch.sum(value >= 0.5).item() / (len(value) + 1e-8)
        else:
            acc = torch.sum(value < 0.5).item() / (len(value) + 1e-8)
        return acc


class DiscriminatorWithOptimizer(Discriminator):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, 
                 in_ch=3, 
                 feat_ch=64,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 logger=None,
                 out_ch = 1,
                ):
        super().__init__(in_ch, feat_ch, out_ch)
        self.optim = optimizer(self.parameters(), 
                               **optim_args)
        if logger is None:
            self.log = {}
            self.logger = lambda x: self.log.update(x)
        else:
            self.logger = logger
    
    def _step(self, x, real:bool):
        self.optim.zero_grad()
        x =  self.forward(x.clone().detach()).reshape(-1, 1)
        target = torch.ones_like(x) if real else torch.zeros_like(x)
        loss = binary_cross_entropy(x, target)
        loss.backward()
        self.optim.step()
        return loss.detach(), x.clone().detach()

    def _log(self, real_loss, fake_loss, real_values, fake_values):
        real_acc = Discriminator._accuracy(real_values, real=True)
        fake_acc = Discriminator._accuracy(fake_values, real=False)
        self.logger({'discriminator_real_loss': real_loss.mean().item(),
                     'discriminator_real_accuracy': real_acc,
                     'discriminator_fake_loss': fake_loss.mean().item(),
                     'discriminator_fake_accuracy': fake_acc}
                    )

    def update(self, real, fake):
        real_loss, real_values = self._step(real, True)
        fake_loss, fake_values = self._step(fake, False)

        self._log(real_loss, fake_loss, real_values, fake_values)
        
        return real_loss + fake_loss


    def _loss(self, real, fake):
        r =  self.forward(real).reshape(-1, 1)
        real_loss = binary_cross_entropy(r, torch.ones_like(r))
        f = self.forward(fake).reshape(-1, 1)
        fake_loss = binary_cross_entropy(f, torch.zeros_like(f))
        self._log(real_loss, fake_loss, r, f)
        return real_loss + fake_loss


class MultiClassDiscriminatorWithOptimizer(DiscriminatorWithOptimizer):
    def __init__(self, 
                 in_ch=3, 
                 feat_ch=64,
                 num_classes=10,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 logger=None
                ):
        super().__init__(in_ch, feat_ch, 
                         optimizer, 
                         optim_args, 
                         logger,
                         out_ch=num_classes,)
        '''Assumes class zero is discarded/background as does MRCNN.'''        


    def _step(self, x, real:bool, classes):
        if len(x) == 0: return torch.empty(0).to(x), torch.empty(0).to(x)
        self.optim.zero_grad()
        x =  self.forward(x.clone().detach(), classes).reshape(-1, 1)
        target = torch.ones_like(x) if real else torch.zeros_like(x)
        loss = binary_cross_entropy(x, target)
        loss.backward()
        self.optim.step()
        return loss.detach(), x.clone().detach()

    def _log(self, real_loss, fake_loss, real_values, fake_values):
        return DiscriminatorWithOptimizer._log(self, real_loss, fake_loss, 
                                               real_values, fake_values)            

    def update(self, real, real_classes, fake, fake_classes):
        real_losses, real_vals = self._step(real, True, real_classes)
        fake_losses, fake_vals = self._step(fake, False, fake_classes)
        self._log(real_losses, fake_losses, real_vals, fake_vals)
        return (real_losses + fake_losses) / 2
    
    def forward(self, x, class_id):
        losses = DiscriminatorWithOptimizer.forward(self, x)
        return losses[:, class_id].flatten() if len(losses) > 0 else torch.tensor([])
    





class MultiDiscriminatorWithOptimizer(nn.Module):
    def __init__(self, 
                 in_ch=3, 
                 feat_ch=64,
                 num_classes=10,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 logger=None
                ):
        super().__init__()
        '''Assumes class zero is discarded/background as does MRCNN.'''
        self.discriminators = nn.ModuleDict({
            str(i): DiscriminatorWithOptimizer(in_ch, 
                                               feat_ch, 
                                               optimizer, 
                                               optim_args, 
                                               logger)
            for i in range(1, num_classes)
        })
        self.logger = logger
        

    def _multihead_step(self, latents, is_real:bool, z_classes):
        losses, values = [], []
        for idx, z in zip(z_classes, latents):
            disc = self.discriminators[str(idx.item())]
            loss, val = disc._step(z[None], is_real)
            
            losses.append(loss.item()); values.append(val)
        return torch.tensor(losses), torch.tensor(values)

    def _log(self, real_loss, fake_loss, real_values, fake_values):
        return DiscriminatorWithOptimizer._log(self, real_loss, fake_loss, 
                                               real_values, fake_values)            

    def update(self, real, real_classes, fake, fake_classes):
        real_losses, real_vals = self._multihead_step(real, 
                                                      True, 
                                                      real_classes)
        fake_losses, fake_vals = self._multihead_step(fake, 
                                                      False, 
                                                      fake_classes)
        
        self._log(real_losses, fake_losses, real_vals, fake_vals)

        return torch.mean(torch.cat([real_losses, fake_losses])).detach()
    
    def forward(self, x, class_id):
        losses = []
        for i, p in zip(class_id, x):
            losses.append(self.discriminators[str(i.item())].forward(p[None]))
        return torch.cat(losses) if len(losses) > 0 else torch.tensor([])
    

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
        super().__init__(in_ch+1, feat_ch, num_classes, optimizer, optim_args, logger)

class ContextAwareDiscriminator(nn.Module):
    '''Discriminator built for FiLM layers. Accepts depth as context.'''
    def __init__(self, in_ch, feat_ch, dim_ctx):
        super().__init__()
        def _f(i, f, k=4, s=2, p=1, act=True, bn=True):  
            '''Helper for layer definition'''
            FiLMLayer(in_ch=i, 
                      out_ch=f,
                      kernel_size=k, 
                      stride=s,
                      padding=p, 
                      bias=False,
                      dim_ctx=dim_ctx,
                      activation=nn.LeakyReLU(0.2, inplace=True) \
                                 if act else lambda x:x,
                      batch_norm=nn.BatchNorm2d \
                                 if bn else lambda x:x
            )
        self._model_list = nn.ModuleList([
            _f(in_ch,     feat_ch,   bn=False     ), # no batchnorm
            _f(feat_ch,   2*feat_ch,              ),
            _f(2*feat_ch, 4*feat_ch,              ),
            _f(4*feat_ch, 1,         k=3, s=1, p=0), # final layer
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x, ctx):
        for m in self._model_list:
            x = m(x, ctx)
        return x
    
