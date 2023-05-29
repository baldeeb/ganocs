import sys
sys.path.insert(0, '/home/baldeeb/Code/pytorch-NOCS')

from models.nocs import get_nocs_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from habitat_datagen_util.utils.dataset import HabitatDataset
from torch.utils.data import DataLoader
from habitat_datagen_util.utils.collate_tools import collate_fn
from torch.optim import Adam
import torch


def targets2device(targets, device):
    for i in range(len(targets)): 
        for k in ['masks', 'labels', 'boxes']: 
            targets[i][k] = targets[i][k].to(device)
    return targets

if __name__ == '__main__':
    DATA_DIR = "/home/baldeeb/Code/pytorch-NOCS/data/habitat/train"  # larger dataset
    device= 'cuda:1'

    model = get_nocs_resnet50_fpn(
                    maskrcnn_weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                    nocs_num_bins=32,
                    nocs_loss_mode = 'classification',
                    multiheaded_nocs = True,

                    )
    model.to(device).train()
    habitatdata = HabitatDataset(DATA_DIR)
    dataloader = DataLoader(habitatdata, 
                            batch_size=2, 
                            shuffle=True, 
                            collate_fn=collate_fn)
    optimizer = Adam(model.parameters())

    model.roi_heads.training_mode('multiview')
    for itr, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets2device(targets, device)
        losses = model(images, targets)
        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(losses['multiview_consistency_loss'])