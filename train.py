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

device='cuda:1'
PATH='/home/baldeeb/Code/pytorch-NOCS/checkpoints/nocs.pth'
# Initialize Model
########################################################################
if True:
    model = get_nocs_resnet50_fpn(maskrcnn_weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
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

habitatdata = HabitatDataloader("/home/baldeeb/Code/pytorch-NOCS/data/habitat-generated/00847-bCPU9suPUw9/metadata.json")
dataloader = DataLoader(habitatdata, batch_size=2, shuffle=True, collate_fn=collate_fn)

def targets2device(targets, device):
    for i in range(len(targets)): 
        for k in ['masks', 'labels', 'boxes']: 
            targets[i][k] = targets[i][k].to(device)
    return targets

wandb.init(project="torch-nocs", name="adding_cls")
optim = Adam(model.parameters(), lr=1e-4)

for epoch in tqdm(range(100)):
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
                nocs = r[0]['nocs']
                wandb.log({
                    'image': wandb.Image(_printable(images[0])),
                    'nocs':wandb.Image(_printable(nocs[0]))
                    })
                model.train()

    torch.save(model.state_dict(), PATH)
    wandb.log({'epoch': epoch})

