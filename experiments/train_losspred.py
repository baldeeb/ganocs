import torch.optim as optim
import torch

from tqdm import tqdm
from time import time
import wandb
from pathlib import Path
import os

from datasets.loss_pred_dataset import NocsLossDataset
from models.nocs_loss_model import Resnet18LossPred


# Configurations
PROJECT_PATH    = Path('/home/baldeeb/Code/pytorch-NOCS/')
EVAL_DATA_PATH  = PROJECT_PATH / Path( 'data/output/1682892735.3506837')
TRAIN_DATA_PATH = PROJECT_PATH / Path( 'data/output/1682896412.9365406')
CHKPT_PATH      = PROJECT_PATH / Path(f'checkpoints/losspred/{time()}')
LOG_NAME        = None # 'largeData_negLogProb'
device          = 'cuda:0'

run_metadata = wandb.Artifact('metadata', 
                              type='run metadata',
                              description='metadata for run',
                              metadata={'train_data_dir' :str(TRAIN_DATA_PATH),
                                        'eval_data_dir'  :str(EVAL_DATA_PATH),
                                        'chkpt_path'     :str(CHKPT_PATH)}
                            )

# Dataset
train_dataloader = NocsLossDataset.get_dataloader(TRAIN_DATA_PATH, 
                                                  64, 'stratified')
eval_dataloader  = NocsLossDataset.get_dataloader(EVAL_DATA_PATH, 64)

# Model
model = Resnet18LossPred().to(device).train() 
optim = optim.Adam(model.parameters(), lr=1e-3)

# Training
if LOG_NAME: 
    wandb.init(project='nocs-loss-pred', name=LOG_NAME)
    wandb.log_artifact(run_metadata)

for epoch in tqdm(range(10)):
    if LOG_NAME: wandb.log({'epoch': epoch})
    for itr, (img, gt) in enumerate(train_dataloader):
        img, gt = img.to(device), gt.to(device)
        loss = model.get_loss(img, gt)
        optim.zero_grad(); loss.backward(); optim.step()
        if LOG_NAME: wandb.log({'loss': loss.item()})    
        if itr % 10 == 0:
            model.eval()
            with torch.no_grad():
                for img, gt in eval_dataloader:
                    img, gt = img.to(device), gt.to(device)
                    loss = model.get_loss(img, gt)
                    if LOG_NAME: wandb.log({'eval_loss': loss.item()})
            model.train()
            
    # Save checkpoint
    os.makedirs(CHKPT_PATH, exist_ok=True)
    torch.save(model.state_dict(), CHKPT_PATH/f'{epoch}.pth')
