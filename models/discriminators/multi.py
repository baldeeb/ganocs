import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy

from .optimizer_wrapper import DiscriminatorWithOptimizer, DiscriminatorWithWessersteinOptimizer

class MultiClassDiscriminatorWithOptimizer(DiscriminatorWithOptimizer):
    def __init__(self, 
                 discriminator,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 'betas':(0.5, 0.999)},
                 logger=None):
        '''Assumes class zero is discarded/background as does MRCNN.'''        
        super().__init__(discriminator, 
                         optimizer, 
                         optim_args, 
                         logger,)

    # def _step(self, x, real:bool, **kwargs):
    #     if len(x) == 0: return torch.empty(0).to(x), torch.empty(0).to(x)
    #     self.optim.zero_grad()
    #     x =  self.forward(x.clone().detach(), **kwargs).reshape(-1, 1)
    #     target = torch.ones_like(x) if real else torch.zeros_like(x)
    #     loss = binary_cross_entropy(x, target)
    #     loss.backward()
    #     self.optim.step()
    #     return loss.detach(), x.clone().detach()

    # def update_real(self, data, **kwargs):
    #     loss, vals = self._step(data, True, **kwargs)
    #     self._log(loss, vals, True)
    #     return loss
    
    # def update_fake(self, data, **kwargs):
    #     loss, vals = self._step(data, False, **kwargs)
    #     self._log(loss, vals, False)
    #     return loss
    
    def forward(self, x, classes):
        losses = super().forward(x)
        losses = losses.gather(-1, classes[:, None]).flatten() 
        return losses if len(losses) > 0 else torch.tensor([])
    
    @property
    def properties(self): return ['multiclass'] + self.discriminator.properties

class MultiClassDiscriminatorWithWessersteinOptimizer(DiscriminatorWithWessersteinOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, classes):
        losses = super().forward(x)
        losses = losses.gather(-1, classes[:, None]).flatten() 
        return losses if len(losses) > 0 else torch.tensor([])
    
    @property
    def properties(self): return ['multiclass'] + super().properties




class MultiDiscriminatorWithWessersteinOptimizer(nn.Module):
    def __init__(self, 
                 discriminators,
                 num_classes=10,
                 optimizer=None,
                 optim_type=DiscriminatorWithWessersteinOptimizer,
                 optim_args=None,
                 logger=None,
                ):
        super().__init__()
        '''Assumes class zero is discarded/background as does MRCNN.'''
        self._optim_wrapper_type = optim_type
        optim_kwargs = {'logger': logger}
        if optim_args is not None: optim_kwargs['optim_args'] = optim_args
        if optimizer is not None: optim_kwargs['optimizer'] = optimizer
        self.discriminators = nn.ModuleDict({
            str(i): optim_type(
                        discriminators[i],
                        **optim_kwargs,
                    )
            for i in range(1, num_classes)
        })
        self.logger = logger

    def _zero_grad(self):
        for d in self.discriminators.values():
            d.zero_grad()
    def _step_optim(self):
        for d in self.discriminators.values():
            d.optim.step()

    def update(self, real_data, fake_data, real_kwargs, fake_kwargs):
        self._zero_grad()
        r =  self.forward(real_data.clone().detach(), **real_kwargs).reshape(-1, 1)
        f =  self.forward(fake_data.clone().detach(), **fake_kwargs).reshape(-1, 1)
        loss = r.mean() - f.mean()
        loss.backward()
        self._step_optim()
        self.logger({'discriminator_real_loss': r.mean().item(),
                    'discriminator_fake_loss': f.mean().item(),
                    'discriminator_loss': loss.item()})
        return loss.detach()

    def critique(self, x, **kwargs):
        return self.forward(x, **kwargs).mean()
    
    def forward(self, x, classes, **kwargs):
        return self.forward_separate(x, classes, **kwargs)
    
    def forward_separate(self, x, classes, **kwargs):
        '''assumes kwargs are lists of the same length as classes'''
        losses = [torch.tensor([]).to(x)]
        for i, (c, p) in enumerate(zip(classes, x)):
            ith_kwargs = {k:v[i:i+1] for k,v in kwargs.items()}
            losses.append(self.discriminators[str(c.item())].forward(p[None], **ith_kwargs))
        return torch.cat(losses)
    
    @property
    def properties(self): 
        return ['multiclass'] + self.discriminators['1'].properties





class MultiDiscriminatorWithOptimizer(nn.Module):
    def __init__(self, 
                 discriminators,
                 num_classes=10,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 optim_type=DiscriminatorWithOptimizer,
                 logger=None,
                ):
        super().__init__()
        '''Assumes class zero is discarded/background as does MRCNN.'''
        self._optim_wrapper_type = optim_type
        self.discriminators = nn.ModuleDict({
            str(i): optim_type(
                        discriminators[i], 
                        optimizer, 
                        optim_args, 
                        logger
                    )
            for i in range(1, num_classes)
        })
        self.logger = logger
        
    def _multihead_step(self, latents, is_real:bool, classes, **kwargs):
        losses, values = [], []

        # TODO: bunch into groups of classes
        # for c in classes.unique():
        #     indices = (classes == c).nonzero(as_tuple=True)[0].clone().detach().long().cpu().numpy()
        #     select_kwargs = {k:v[indices] for k,v in kwargs.items()}
        #     select_latents = latents[indices]
        #     disc = self.discriminators[str(c.item())]
        #     loss, val = disc._step(select_latents, is_real, **select_kwargs)
        #     losses.append(loss.item()); values.append(val)

        for i, (c, z) in enumerate(zip(classes, latents)):
            ith_kwargs = {k:v[i:i+1] for k,v in kwargs.items()}
            disc = self.discriminators[str(c.item())]
            loss, val = disc._step(z[None], is_real, **ith_kwargs)
            losses.append(loss.item()); values.append(val)
        return torch.tensor(losses), torch.tensor(values)

    def _log(self, *args, **kwargs):
        return self._optim_wrapper_type._log(self, *args, **kwargs)            

    def _update(self, data, real, classes, **kwargs):
        loss, val = self._multihead_step(data, real, classes, **kwargs)
        self._log(loss, val, real)
        return loss.mean().detach()
    
    def update(self, real_data, fake_data, real_kwargs, fake_kwargs):
        r = self._update(real_data.clone().detach(), True, **real_kwargs)
        f = self._update(fake_data.clone().detach(), False, **fake_kwargs)
        return (r + f) / 2
    # def update_real(self, data, classes, **kwargs):
    #     return self._update(data, True, classes, **kwargs)
    # def update_fake(self, data, classes, **kwargs):
    #     return self._update(data, False, classes, **kwargs)

    def forward(self, x, classes, **kwargs):
        # return self.forward_all(x, classes, **kwargs)
        return self.forward_separate(x, classes, **kwargs)
    
    def forward_separate(self, x, classes, **kwargs):
        '''assumes kwargs are lists of the same length as classes'''
        losses = []
        for i, (c, p) in enumerate(zip(classes, x)):
            ith_kwargs = {k:v[i:i+1] for k,v in kwargs.items()}
            losses.append(self.discriminators[str(c.item())].forward(p[None], **ith_kwargs))
        return torch.cat(losses) if len(losses) > 0 else torch.tensor([])


    def forward_all(self, x, classes, **kwargs):
        '''assumes kwargs are lists of the same length as classes'''
        losses = []
        for i, (k, d) in enumerate(self.discriminators.items()):
            ith_kwargs = {k:v[i:i+1] for k,v in kwargs.items()}
            mask = classes == int(k)
            losses.extend(d(x, **ith_kwargs)[mask])
        return torch.cat(losses) if len(losses) > 0 else torch.tensor([])

    @property
    def properties(self): 
        return ['multiclass'] + self.discriminators['1'].properties
