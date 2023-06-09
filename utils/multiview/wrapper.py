
from torch.utils.data import DataLoader
import torch

class MultiviewLossFunctor:
    '''This wrapper employs some hacks to zero in on the multiview loss.
    Ideally, the model would allow us to run data withouth the need to
    pass target data and while avoiding other loss calculations.
    
    TODO: try to fix that...
    
    '''
    def __init__(self, 
                 dataloader:DataLoader, 
                 model:torch.nn.Module, 
                 weight:float, 
                 mode:str,
                 device:str):
        self.dataloader = dataloader
        self.data_iter = self.dataloader.__iter__()
        self.model = model
        self.weight = weight
        self.device = device

        self.model.roi_heads._multiview_loss_mode = mode # TODO: fix / organize


    def _get_data(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            print('WARNING: Multiview dataloader ran out of data. Resetting.')  # TODO: log this in a better way
            self.data_iter = self.dataloader.__iter__()
            return next(self.data_iter)

    @staticmethod
    def _targets2device(targets, device):
            for i in range(len(targets)): 
                for k in ['masks', 'labels', 'boxes']: 
                    targets[i][k] = targets[i][k].to(device)
            return targets
    
    def __call__(self):
        init_training_mode = self.model.roi_heads._training_mode
        self.model.roi_heads.training_mode('multiview')
        images, targets = self._get_data()
        images = images.to(self.device)
        targets = self._targets2device(targets, self.device)
        losses = self.model(images, targets)
        self.model.roi_heads.training_mode(init_training_mode)
        
        losses = {k: v * self.weight for k, v in losses.items() if 'multiview' in k}
        return losses


