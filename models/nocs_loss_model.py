import torchvision.models as models
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
import torch

class Resnet18LossPred(nn.Module):
    '''
    Model adapted to predict the loss of a given NOCS image.

    NOTE: The model was tested with and without LogSigmoid. 
        The performance was comparable but theoretically the
        log of sigmoid is more sound.
    '''

    def __init__(self,):
        super().__init__()
        self.model = models.resnet18()
        self.model.fc = nn.Sequential(nn.Linear(512, 1), 
                                      nn.LogSigmoid())
    
    def forward(self, x): 
        return -self.model(x)

    def get_loss(self, x, gt):
        pred = self.forward(x)
        return F.mse_loss(pred, gt.squeeze())

    @staticmethod
    def load(path):
        model = Resnet18LossPred()
        model.load_state_dict(torch.load(path))
        return model
    
    def fix(self, img):
        '''Given a NOCS image, this function fixes it by
        optimizing the image over using the class model
        as the loss function.'''
        img = img.clone().detach().requires_grad_(True)
        optim = Adam([img], lr=1e-1)
        for _ in range(10):
            loss = self.forward(img)
            optim.zero_grad(); 
            loss.backward(); 
            optim.step()
        return img