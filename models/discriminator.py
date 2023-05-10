import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy

'''
    
NOTE s:
    - using a lr of 1e-4 for the discriminator
        -> the generator loss to explode. 
        -> Not good.
    - setting the lr of the discriminator to 1e-5
        -> discriminator loss to converge and plateau around 0.8. 
        -> Not good.
    - lr=2e-4 and normal initialization of weights. 
        -> discriminator too good.
        -> generator loss explodes.
        -> Not good.
    - lr=5e-5 and normal initialization of weights.
        -> takes longer for the discriminator to dominate.
        -> generator loss explodes.
        -> Not good.
    - lr=2e-5 and normal initialization of weights.   
    
'''


class Discriminator(nn.Module):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, in_ch, feat_ch=128):
        super().__init__()

        # a 4x4 kernel model. closer to DCGAN.
        # NOTE: currently collapses the generator.
        if True:
            fcs = [in_ch, feat_ch, 2*feat_ch, 4*feat_ch, 8*feat_ch, 1]
            self.model = nn.Sequential(
                nn.Conv2d(fcs[0], fcs[1], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[1], fcs[2], 4, 2, 1, bias=False),
                nn.BatchNorm2d(fcs[2]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[2], fcs[3], 4, 2, 1, bias=False),
                nn.BatchNorm2d(fcs[3]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[3], fcs[5], 3, 1, 0, bias=False),
                nn.Sigmoid()
            )
        if False:
            fcs = [in_ch, feat_ch, feat_ch, feat_ch, feat_ch, 1]
            self.model = nn.Sequential(
                nn.Conv2d(fcs[0], fcs[1], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[1], fcs[2], 4, 2, 1, bias=False),
                nn.BatchNorm2d(fcs[2]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[2], fcs[3], 4, 2, 1, bias=False),
                nn.BatchNorm2d(fcs[3]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[3], fcs[5], 3, 1, 0, bias=False),
                nn.Sigmoid()
            )
        if False:
            # a 3x3 kernel model.
            fcs = [in_ch, feat_ch, 2*feat_ch, 4*feat_ch, 8*feat_ch, 1]
            self.model = nn.Sequential(
                nn.Conv2d(fcs[0], fcs[1], 3, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[1], fcs[2], 3, 2, 1, bias=False),
                nn.BatchNorm2d(fcs[2]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[2], fcs[3], 3, 2, 1, bias=False),
                nn.BatchNorm2d(fcs[3]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(fcs[3], fcs[5], 3, 1, 0, bias=False),
                nn.Sigmoid()
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x):
        return self.model(x)
    
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
        # self.optim.zero_grad()
        # loss = self._loss(real, fake)
        # loss.backward(retain_graph=True)
        # self.optim.step()
        # return loss.item()
        
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
