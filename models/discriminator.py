import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from models.film import FiLMLayer


class Discriminator(nn.Module):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, in_ch, feat_ch=128):
        super().__init__()

        # a 4x4 kernel model. closer to DCGAN.
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_ch,   2*feat_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(2*feat_ch, 4*feat_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv2d(4*feat_ch, 8*feat_ch, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(8*feat_ch),
            # nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(4*feat_ch, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x):
        return self.model(x)
    
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
            x = m (x, ctx)
        return x

class DiscriminatorWithOptimizer(Discriminator):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, 
                 in_ch=3, 
                 feat_ch=64,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 logger=None
                ):
        super().__init__(in_ch, feat_ch)
        self.optim = optimizer(self.parameters(), 
                               **optim_args)
        if logger is None:
            self.log = {}
            self.logger = lambda x: self.log.update(x)
    
    def _step(self, x, real:bool):
        self.optim.zero_grad()
        x =  self.forward(x.clone().detach()).reshape(-1, 1)
        target = torch.ones_like(x) if real else torch.zeros_like(x)
        loss = binary_cross_entropy(x, target)
        loss.backward()
        self.optim.step()
        return loss.item(), x.clone().detach()

    def update(self, real, fake):
        real_loss, real_values = self._step(real, True)
        fake_loss, fake_values = self._step(fake, False)

        acc = self.accuracy(real_values, fake_values)
        self.logger({'discriminator_real_loss': real_loss,
                     'discriminator_fake_loss': fake_loss,
                     'discriminator_accuracy': acc})
        return real_loss + fake_loss

    def accuracy(self, real, fake):
        r = torch.sum(real > 0.5).item()
        f = torch.sum(fake < 0.5).item()
        return (r + f) / (real.shape[0] + fake.shape[0])    

    def _loss(self, real, fake):
        r =  self.forward(real).reshape(-1, 1)
        real_loss = binary_cross_entropy(r, torch.ones_like(r))
        f = self.forward(fake).reshape(-1, 1)
        fake_loss = binary_cross_entropy(f, torch.zeros_like(f))
        self.logger({'discriminator_real_loss': real_loss.item(),
                     'discriminator_fake_loss': fake_loss.item(),
                     'discriminator_accuracy': self.accuracy(r, f)
                     })
        return real_loss + fake_loss

class MultiClassDiscriminatorWithOptimizer(nn.Module):
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

    def update(self, targets, predictions, class_id):
        losses = []
        for i, t, p in zip(torch.cat(class_id), targets, predictions):
            losses.append(self.discriminators[str(i.item())].update(t[None], p[None]))
        return torch.mean(torch.tensor(losses))
    
    def forward(self, x, class_id):
        losses = []
        for i, p in zip(torch.cat(class_id), x):
            losses.append(self.discriminators[str(i.item())].forward(p[None]))
        return torch.cat(losses)