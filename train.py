from models.nocs import get_nocs_resnet50_fpn
from models.discriminator import DiscriminatorWithOptimizer
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

from habitat_data_util.utils.dataset import HabitatDataloader
from torch.utils.data import DataLoader
from utils.dataset import collate_fn
from utils.nocs_cache import SaveCachedNocsAndLoss

from torch.optim import Adam
import torch 

from tqdm import tqdm
import wandb

import pathlib as pl
import os
from time import time

# DATA_DIR = "/home/baldeeb/Code/pytorch-NOCS/data/habitat-generated/00847-bCPU9suPUw9/metadata.json"
DATA_DIR = "/home/baldeeb/Code/pytorch-NOCS/data/habitat-generated/200of100scenes_26selectChairs"  # larger dataset

NOCS_SAVE_PATH = f'/home/baldeeb/Code/pytorch-NOCS/data/output/{time()}'
save_nocs_and_loss = SaveCachedNocsAndLoss(NOCS_SAVE_PATH)

device='cuda:0'
CHKPT_PATH = pl.Path(f'/home/baldeeb/Code/pytorch-NOCS/checkpoints/nocs/{time()}')
os.makedirs(CHKPT_PATH)

CACHE = False

# Initialize Model
nocs_discriminator = DiscriminatorWithOptimizer(3)
model = get_nocs_resnet50_fpn(maskrcnn_weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                cache_results=CACHE,
                                nocs_loss=nocs_discriminator)
model.to(device).train()


habitatdata = HabitatDataloader(DATA_DIR)
dataloader = DataLoader(habitatdata, 
                        batch_size=4, 
                        shuffle=True, 
                        collate_fn=collate_fn)

def targets2device(targets, device):
    for i in range(len(targets)): 
        for k in ['masks', 'labels', 'boxes']: 
            targets[i][k] = targets[i][k].to(device)
    return targets

wandb.init(project="torch-nocs", name="largeData_conciseDiscriminator")
# optim = Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))

# NOTE: this is needed or else the discriminator parameters 
#       will get optimizd wih the generator
# model_params = [p for n, p in model.named_parameters() if '_nocs_loss' not in n]
# optim = Adam(model_params, lr=1e-4, betas=(0.5, 0.999))


def run_training(num_epochs):
    for epoch in tqdm(range(num_epochs)):
        for itr, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets2device(targets, device)
            losses = model(images, targets)
            loss = sum(losses.values())
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            wandb.log(losses)
            wandb.log(nocs_discriminator.log)
            wandb.log({'loss': loss})

            with torch.no_grad():
                if itr % 10 == 0:
                    model.eval()
                    r = model(images)
                    # TEMPORARY HACK ###################################################
                    # code is breaking because r is empty sometimes
                    temp = [v.numel() for k, v in r[0].items()]
                    if sum(temp) != 0:
                    ####################################################################
                        _printable = lambda a: a.permute(1,2,0).detach().cpu().numpy()
                        wandb.log({'image': wandb.Image(_printable(images[0])),
                                'nocs':wandb.Image(_printable(r[0]['nocs'][0]))})
                    ####################################################################
                    model.train()

            if CACHE: save_nocs_and_loss(epoch, model.cache)
            if itr % 10 == 0:
                if model.cache:
                    _printable = lambda a: a.permute(1,2,0).detach().cpu().numpy()
                    wandb.log({'cached': wandb.Image(_printable(model.cache[0]['nocs'][0]))})

        torch.save(model.state_dict(), CHKPT_PATH/f'{epoch}.pth')
        wandb.log({'epoch': epoch})

    if CACHE: save_nocs_and_loss.save_losses()


def params_generator(model, ignore=None, select=None):
    for name, param in model.named_parameters():
        if ignore and any([x in name for x in ignore]):
            continue
        if select and not any([x in name for x in select]):
            continue
        yield param

# Freeze nocs heads and discriminator
nocs_discriminator.freeze = True
params = list(params_generator(model, ignore=['nocs_loss', 'nocs_heads']))
optim = Adam(params, lr=1e-4, betas=(0.5, 0.999))
run_training(num_epochs=2)

# only train nocs
nocs_discriminator.freeze = False
params = list(params_generator(model, select=['nocs_loss', 'nocs_heads']))
optim = Adam(params, lr=2e-4, betas=(0.5, 0.999))
run_training(num_epochs=10)