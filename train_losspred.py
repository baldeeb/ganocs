import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch

from tqdm import tqdm
from time import time
import wandb
from pathlib import Path
import os
from dataset import NocsLossDataset
from torch.utils.data import DataLoader, WeightedRandomSampler

# Configurations
PROJECT_PATH    = Path('/home/baldeeb/Code/pytorch-NOCS/')
EVAL_DATA_PATH  = PROJECT_PATH / Path( 'data/output/1682892735.3506837')
TRAIN_DATA_PATH = PROJECT_PATH / Path( 'data/output/1682896412.9365406')
CHKPT_PATH      = PROJECT_PATH / Path(f'checkpoints/losspred/{time()}')
LOG_NAME        = 'largeData_negLogProb'
device          = 'cuda:1'
os.makedirs(CHKPT_PATH, exist_ok=True)

run_metadata = wandb.Artifact('metadata', 
                              type='run metadata',
                              description='metadata for run',
                              metadata={'train_data_dir' :str(TRAIN_DATA_PATH),
                                        'eval_data_dir'  :str(EVAL_DATA_PATH),
                                        'chkpt_path'     :str(CHKPT_PATH)}
                            )

# Data loading
def get_dataloader(data_dir, 
                   batch_size=64, 
                   sampling=None):
    dataset = NocsLossDataset(data_dir)
    if sampling == 'stratified':
        weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, 
                                        len(weights))
    else: sampler = None
    return DataLoader(dataset, sampler=sampler, 
                      batch_size=batch_size)

train_dataloader = get_dataloader(TRAIN_DATA_PATH, 64, 'stratified')
eval_dataloader  = get_dataloader(EVAL_DATA_PATH, 64)

# Model definition
class LossPred(nn.Module):
    '''
    Model adapted to predict the loss of a given NOCS image.

    NOTE: The model was tested with and without LogSigmoid. 
        The performance was comparable but theoretically the
        log of sigmoid is more sound.
    '''

    def __init__(self,):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(nn.Linear(512, 1), nn.LogSigmoid())
    
    def forward(self, x): 
        return -self.model(x)

    @staticmethod
    def load(path):
        model = LossPred()
        model.load_state_dict(torch.load(path))
        return model

def visualize(img, loss):
    import matplotlib.pyplot as plt
    plt.imshow(img.permute(1,2,0).detach().cpu().numpy())
    plt.title(f'Loss: {loss.item()}')
    plt.show()

model = LossPred().to(device)
model.train()    

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training
if LOG_NAME: 
    wandb.init(project='nocs-loss-pred', name=LOG_NAME)
    wandb.log_artifact(run_metadata)

for epoch in tqdm(range(10)):
    for itr, (img, gt) in enumerate(train_dataloader):
        img, gt = img.to(device), gt.to(device)
        pred = model(img).squeeze()
        loss = criterion(pred, gt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if LOG_NAME: wandb.log({'loss': loss.item()})    
        if itr % 10 == 0:
            model.eval()
            with torch.no_grad():
                for img, gt in eval_dataloader:
                    img, gt = img.to(device), gt.to(device)
                    pred = model(img).squeeze()
                    loss = criterion(pred, gt)
                    if LOG_NAME: wandb.log({'eval_loss': loss.item()})
            model.train()
    if LOG_NAME: wandb.log({'epoch': epoch})
    torch.save(model.state_dict(), CHKPT_PATH/f'{epoch}.pth')
