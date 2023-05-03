from models.nocs import NOCS, get_nocs_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone    
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

from habitat_data_util.utils.dataset import HabitatDataloader
from torch.utils.data import DataLoader
from utils.dataset import collate_fn

from torch.optim import Adam
import torch 

from tqdm import tqdm
import wandb

import pathlib as pl
import os
from time import time


import cv2
import numpy as np
import json
class SaveCachedNocsAndLoss():
    def __init__(self, out_folder):
        self._folder = out_folder
        self._losses = {}
    def __call__(self, epoch, cache):
        for i, result in enumerate(cache):
            for j, nocs in enumerate(result['nocs']):
                f_name = f'{epoch}_{time()}_nocs.png'
                img = nocs.permute(1,2,0).detach().cpu().numpy()
                cv2.imwrite(f'{self._folder}/{f_name}', 
                            (img*255).astype(np.uint8))

                self._losses[f_name] = result['loss_nocs'][j].numpy().tolist()
    
    def save_losses(self):
        f = open(f'{self._folder}/losses.json', 'w+')
        json.dump(self._losses, f)
        f.close()

NOCS_SAVE_PATH = f'/home/baldeeb/Code/pytorch-NOCS/data/output/{time()}'
os.makedirs(NOCS_SAVE_PATH)
save_nocs_and_loss = SaveCachedNocsAndLoss(NOCS_SAVE_PATH)


device='cuda:0'
CHKPT_PATH = pl.Path(f'/home/baldeeb/Code/pytorch-NOCS/checkpoints/nocs/{time()}')
os.makedirs(CHKPT_PATH)
# Initialize Model
########################################################################
if True:
    model = get_nocs_resnet50_fpn(maskrcnn_weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                  cache_results=True)
    model.to(device).train()
else:
    # NOTE: this is temporary test to make sure MaskRCNN works
    # This helps debug the dataloader 
    from torchvision.models.detection import maskrcnn_resnet50_fpn 
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = maskrcnn_resnet50_fpn(pretrained=True)  # load an instance segmentation model pre-trained pre-trained on COCO
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
    model.to(device)# move model to the right device
    model.train()
########################################################################

# habitatdata = HabitatDataloader("/home/baldeeb/Code/pytorch-NOCS/data/habitat-generated/00847-bCPU9suPUw9/metadata.json")
habitatdata = HabitatDataloader("/home/baldeeb/Code/pytorch-NOCS/data/habitat-generated/200of100scenes_26selectChairs")
dataloader = DataLoader(habitatdata, batch_size=8, shuffle=True, collate_fn=collate_fn)

def targets2device(targets, device):
    for i in range(len(targets)): 
        for k in ['masks', 'labels', 'boxes']: 
            targets[i][k] = targets[i][k].to(device)
    return targets

wandb.init(project="torch-nocs", name="caching-implemented")
optim = Adam(model.parameters(), lr=1e-4)

for epoch in tqdm(range(15)):
    for itr, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets2device(targets, device)
        losses = model(images, targets)
        loss = sum(losses.values())
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        wandb.log(losses)
        wandb.log({'loss': loss})

        with torch.no_grad():
            if itr % 10 == 0:
                model.eval()
                r = model(images)
                _printable = lambda a: a.permute(1,2,0).detach().cpu().numpy()
                wandb.log({
                    'image': wandb.Image(_printable(images[0])),
                    'nocs':wandb.Image(_printable(r[0]['nocs'][0]))
                    })
                model.train()

        save_nocs_and_loss(epoch, model.cache)
        if itr % 10 == 0:
            if model.cache:
                _printable = lambda a: a.permute(1,2,0).detach().cpu().numpy()
                wandb.log({'cached': wandb.Image(_printable(model.cache[0]['nocs'][0]))})

    torch.save(model.state_dict(), CHKPT_PATH/f'{epoch}.pth')
    wandb.log({'epoch': epoch})

save_nocs_and_loss.save_losses()