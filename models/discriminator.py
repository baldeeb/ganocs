import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy

class Discriminator(nn.Module):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, in_ch, feat_ch=64):
        super().__init__()
        fcs = [in_ch, feat_ch, 2*feat_ch, 
               4*feat_ch, 8*feat_ch, 1]
        self.model = nn.Sequential(
            nn.Conv2d(fcs[0], fcs[1], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fcs[1], fcs[2], 4, 2, 1),
            nn.BatchNorm2d(fcs[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(fcs[2], fcs[3], 4, 2, 1),
            # nn.BatchNorm2d(fcs[3]),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(fcs[3], fcs[4], 4, 2, 1),
            # nn.BatchNorm2d(fcs[4]),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(fcs[4], fcs[5], 4, 1, 0),
            # nn.Sigmoid()

            nn.Conv2d(fcs[2], fcs[3], 4, 2, 2),
            nn.BatchNorm2d(fcs[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fcs[3], fcs[5], 4, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
    
class DiscriminatorWithOptimizer(Discriminator):
    '''Discriminator model for NOCS images.
    ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def __init__(self, 
                 in_ch, 
                 feat_ch=64,
                 optimizer=torch.optim.Adam,
                 optim_args={'lr':1e-4, 
                             'betas':(0.5, 0.999)},
                 logger=lambda *_: None
                ):
        super().__init__(in_ch, feat_ch)
        self.optim = optimizer(self.parameters(), 
                               **optim_args)
        self.logger = logger
    
    def train_step(self, real, fake):
        self.optim.zero_grad()
        loss = self._loss(real, fake)
        loss.backward(retain_graph=True)
        self.optim.step()
        return loss.item()
    
    def _loss(self, real, fake):
        r =  self.forward(real).reshape(-1, 1)
        real_loss = binary_cross_entropy(r, torch.ones_like(r))
        f = self.forward(fake).reshape(-1, 1)
        fake_loss = binary_cross_entropy(f, torch.zeros_like(f))
        self.logger({'discriminator_real_loss': real_loss.item(),
                     'discriminator_fake_loss': fake_loss.item()})
        return real_loss + fake_loss
