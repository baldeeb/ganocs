import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy

from .util import accuracy

class DiscriminatorWithOptimizer(nn.Module):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, 
                 discriminator,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 'betas':(0.5, 0.999)},
                 logger=None, ):
        super().__init__()
        self.discriminator = discriminator
        self.optim = optimizer(self.parameters(), 
                               **optim_args)
        if logger is None:
            self.log = {}
            self.logger = lambda x: self.log.update(x)
        else:
            self.logger = logger
    
    def forward(self, x, **kwargs): 
        return self.discriminator(x, **kwargs)

    def _step(self, x, real:bool, **kwargs):
        self.optim.zero_grad()
        x =  self.forward(x.clone().detach(), **kwargs).reshape(-1, 1)
        target = torch.ones_like(x) if real else torch.zeros_like(x)
        loss = binary_cross_entropy(x, target)
        loss.backward()
        self.optim.step()
        return loss.detach(), x.clone().detach()
    
    def _log(self, losses, values, real:bool):
        name = 'real' if real else 'fake'
        acc = accuracy(values, real=real)
        self.logger({f'discriminator_{name}_loss': losses.mean().item(),
                     f'discriminator_{name}_accuracy': acc,})

    def _update(self, data, real:bool, **kwargs):
        loss, val = self._step(data, real, **kwargs)
        self._log(loss, val, real=real)
        return loss
    
    def update_real(self, data, **kwargs): 
        return self._update(data, True,  **kwargs)
    def update_fake(self, data, **kwargs): 
        return self._update(data, False, **kwargs)
    
    def update(self, real_data, fake_data, real_kwargs, fake_kwargs):
        r = self.update_real(real_data.clone().detach(), **real_kwargs)
        f = self.update_fake(fake_data.clone().detach(), **fake_kwargs)
        return (r + f) / 2

    def _loss(self, real, fake):
        r =  self.forward(real).reshape(-1, 1)
        real_loss = binary_cross_entropy(r, torch.ones_like(r))
        self._log(real_loss, r, True)
        f = self.forward(fake).reshape(-1, 1)
        fake_loss = binary_cross_entropy(f, torch.zeros_like(f))
        self._log(fake_loss, f, False)
        return real_loss + fake_loss
    
    def critique(self, x, reduction='mean', **kwargs):
        l = self.forward(x, **kwargs)
        return binary_cross_entropy(l, torch.ones_like(l), 
                                    reduction=reduction)

    @property
    def properties(self): 
        return ['with_optimizer'] + self.discriminator.properties
    

class DiscriminatorWithWessersteinOptimizer(nn.Module):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, 
                 discriminator,
                 optimizer=torch.optim.RMSprop,
                 optim_args={'lr':5e-5, 'alpha':0.99, 'eps':1e-8},
                 logger=lambda _: None,
                ):
        super().__init__()
        self.discriminator = discriminator
        self.optim = optimizer(self.parameters(), **optim_args)
        self.logger = logger
    
    def forward(self, x, **kwargs): 
        return self.discriminator(x, **kwargs)

    def update(self, real_data, fake_data, real_kwargs, fake_kwargs, **kwargs):
        self.optim.zero_grad()
        
        r =  self.forward(real_data.clone().detach(), **real_kwargs).reshape(-1, 1)
        f =  self.forward(fake_data.clone().detach(), **fake_kwargs).reshape(-1, 1)
        loss = r.mean() - f.mean()
        loss.backward()
        self.optim.step()
        self.logger({'discriminator_real_loss': r.mean().item(),
                    'discriminator_fake_loss': f.mean().item(),
                    'discriminator_loss': loss.item()})
        return loss.detach()

    def critique(self, x, **kwargs):
        return self.forward(x, **kwargs).mean()

    @property
    def properties(self): 
        return ['with_optimizer'] + self.discriminator.properties