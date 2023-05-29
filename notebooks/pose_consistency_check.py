import sys
sys.path.insert(0, '/home/baldeeb/Code/pytorch-NOCS')

from models.nocs import get_nocs_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from utils.load_save import load_nocs
from torch.optim import Adam
import torch

from habitat_datagen_util.utils.dataset import HabitatDataset
from habitat_datagen_util.utils.collate_tools import collate_fn

from utils.spot_dataset import SpotDataset, collate_fn 

def targets2device(targets, device):
    for i in range(len(targets)): 
        for k in ['masks', 'labels', 'boxes']: 
            targets[i][k] = targets[i][k].to(device)
    return targets

def habitat_run():
    # DATA_DIR = "/home/baldeeb/Code/pytorch-NOCS/data/habitat/train"  # larger dataset
    DATA_DIR = '/home/baldeeb/Code/pytorch-NOCS/data/habitat/generated/200of100scenes_26selectChairs/test'
    CHKPT_DIR = "/home/baldeeb/Code/pytorch-NOCS/checkpoints/nocs_classification/2023-05-25_13-46-18/one_class_frozen_backbone_4.pth"
    device= 'cuda:1'

    # model = get_nocs_resnet50_fpn(
    #                 maskrcnn_weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
    #                 nocs_num_bins=32,
    #                 nocs_loss_mode = 'classification',
    #                 multiheaded_nocs = True,
                    # )
    model = load_nocs(CHKPT_DIR,
                    maskrcnn_weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                    nocs_num_bins=32,
                    nocs_loss_mode = 'classification',
                    multiheaded_nocs = True,)
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
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        print(losses['multiview_consistency_loss'])

def spot_run():
    DATA_DIR = '/home/baldeeb/Code/bd_spot_wrapper/data/output/front_left'
    CHKPT_DIR = "/home/baldeeb/Code/pytorch-NOCS/checkpoints/nocs_classification/2023-05-25_13-46-18/one_class_frozen_backbone_4.pth"
    device= 'cuda:1'

    model = load_nocs(CHKPT_DIR,
                    maskrcnn_weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                    nocs_num_bins=32,
                    nocs_loss_mode = 'classification',
                    multiheaded_nocs = True,)
    model.to(device).train()

    habitatdata = SpotDataset(DATA_DIR)
    dataloader = DataLoader(habitatdata, 
                            batch_size=2, 
                            collate_fn=collate_fn)
    optimizer = Adam(model.parameters(keys=['nocs']))

    model.roi_heads.training_mode('multiview')
    for itr, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        # targets = targets2device(targets, device)
        losses = model(images, targets)
        loss = sum(losses.values())
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        print(losses['multiview_consistency_loss'])


if __name__ == '__main__':
    # habitat_run()
    spot_run()