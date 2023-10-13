import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy

from .optimizer_wrapper import DiscriminatorWithOptimizer


class MultiClassDiscriminatorWithOptimizer(DiscriminatorWithOptimizer):
    def __init__(self, 
                 in_ch=3, 
                 feat_ch=64,
                 num_classes=10,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 logger=None):
        '''Assumes class zero is discarded/background as does MRCNN.'''        
        super().__init__(in_ch, feat_ch, 
                         optimizer, 
                         optim_args, 
                         logger,
                         out_ch=num_classes,)

    def _step(self, x, real:bool, **kwargs):
        if len(x) == 0: return torch.empty(0).to(x), torch.empty(0).to(x)
        self.optim.zero_grad()
        x =  self.forward(x.clone().detach(), kwargs).reshape(-1, 1)
        target = torch.ones_like(x) if real else torch.zeros_like(x)
        loss = binary_cross_entropy(x, target)
        loss.backward()
        self.optim.step()
        return loss.detach(), x.clone().detach()

    def update_real(self, data, **kwargs):
        loss, vals = self._step(data, True, **kwargs)
        self._log(loss, vals, True)
        return loss
    
    def update_fake(self, data, **kwargs):
        loss, vals = self._step(data, False, **kwargs)
        self._log(loss, vals, False)
        return loss
    
    def forward(self, x, classes):
        losses = DiscriminatorWithOptimizer.forward(self, x)
        return losses[:, classes].flatten() if len(losses) > 0 else torch.tensor([])
    
    @property
    def properties(self): return ['multiclass'] + self.discriminators['1'].properties


class MultiDiscriminatorWithOptimizer(nn.Module):
    def __init__(self, 
                 discriminators,
                 num_classes=10,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 logger=None
                ):
        super().__init__()
        '''Assumes class zero is discarded/background as does MRCNN.'''
        self.discriminators = nn.ModuleDict({
            str(i): DiscriminatorWithOptimizer(
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
        return DiscriminatorWithOptimizer._log(self, *args, **kwargs)            

    def _update(self, data, real, classes, **kwargs):
        loss, val = self._multihead_step(data, real, classes, **kwargs)
        self._log(loss, val, real)
        return loss.mean().detach()
    
    def update_real(self, data, classes, **kwargs):
        return self._update(data, True, classes, **kwargs)
    
    def update_fake(self, data, classes, **kwargs):
        return self._update(data, False, classes, **kwargs)

    def forward(self, x, classes, **kwargs):
        '''assumes kwargs are lists of the same length as classes'''
        losses = []


        for i, (c, p) in enumerate(zip(classes, x)):
            ith_kwargs = {k:v[i:i+1] for k,v in kwargs.items()}
            losses.append(self.discriminators[str(c.item())].forward(p[None], **ith_kwargs))
        return torch.cat(losses) if len(losses) > 0 else torch.tensor([])

    @property
    def properties(self): 
        return ['multihead'] + self.discriminators['1'].properties
