
import torch
import torchvision
from pathlib import Path
from torchvision.transforms import (
    RandomCrop,
    Normalize, 
    ToTensor,
    RandomHorizontalFlip, 
    Compose
)
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone    

from torch.utils.data import DataLoader

from models.nocs import NOCS
from torchvision.models.detection.mask_rcnn import MaskRCNN
from pycocotools.coco import COCO 

from models.nocs import get_nocs_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights

def get_data():
    # Data directory
    # ## Laptop
    # DATA_FOLDER = Path(__file__).resolve().parent / 'data'
    # IMG_FOLDER = Path('train')
    # ANNOTATION_FOLDER = Path('annotations_trainval2017/annotations/instances_train2017.json')
    ## Lab PC
    DATA_FOLDER = Path('/home/baldeeb/Data/cocodataset/')
    IMG_FOLDER = Path('val2017')
    ANNOTATION_FOLDER = Path('annotations_trainval2017/annotations/instances_val2017.json')
    
    # Data
    dataset = torchvision.datasets.CocoDetection(
                                DATA_FOLDER/IMG_FOLDER
                                ,DATA_FOLDER/ANNOTATION_FOLDER
                                ,transform=ToTensor())
    coco = COCO(DATA_FOLDER/ANNOTATION_FOLDER)
    return dataset, coco


def show_image(im, show=True):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(im)
    if show: plt.show()

def get_boxes(targets):
    boxes = []
    for t in targets:
        boxes.append(t['bbox'])
    return torch.tensor(boxes)

def test_inference():
    # Initialize Model
    m = get_nocs_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    m.eval()

    # Load data
    dataset, _ = get_data()

    # Run loop
    for images, targets in dataset:
        print(f'images shape: {images.shape}, type: {type(images)}, dtype: {images.dtype}')
        predictions = m([images])

        # Display the predicted nocs map
        show_image(images[0], show=False)
        show_image(predictions[0]['nocs'].sum(dim=0).permute(1,2,0).detach().numpy())
        print('done')
        break

def test_loss():
    from models.nocs_loss import nocs_loss
    import matplotlib.pyplot as plt
    import numpy as np

    def show_all_masks(targets):
        mask = coco.annToMask(targets[0])
        for i in range(len(targets)):
            mask += coco.annToMask(targets[i])
        show_image(mask)
        

    # get dataset 
    dataset, coco = get_data()
    # Run loop
    for image, targets in dataset:
        # Get the ground truth nocs map
        print(targets[0].keys())

        # mask = np.stack([coco.annToMask(t) for t in targets])
        mask = np.stack([coco.annToMask(targets[0]), coco.annToMask(targets[1])])
        mask = torch.tensor(mask).unsqueeze(0)

        gt_nocs = torch.tensor(image).unsqueeze(0) # Temporarily since the data does not have NOCS
        gt_mask = mask
        # print(gt_nocs.shape)
        # print(f'mask len {len(gt_mask[0])}')
        # print(gt_mask[0])
        pred_nocs = torch.rand(gt_nocs.shape[0], gt_nocs.shape[-3], mask.shape[1], gt_nocs.shape[-2], gt_nocs.shape[-1])
        pred_nocs = pred_nocs * gt_mask.unsqueeze(1)

        loss = nocs_loss(gt_mask, gt_nocs, pred_nocs)
        print(loss)
        break

def test_training(): 
    # Initialize Model
    m = get_nocs_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    m.train()

    # Load data
    dataset, _ = get_data()

    # Run loop
    for image, targets in dataset:
        for i in range(len(targets)): # Temp Set mock nocs
            targets[i]['nocs'] = torch.tensor(image).unsqueeze(0)
        predictions = m([image], targets)


        # Display the predicted nocs map
        show_image(predictions[0]['nocs'].sum(dim=0).permute(1,2,0).detach().numpy())
        print('done')
        break



def load_pretrained_maskrcnn():
    model = get_nocs_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()

if __name__=='__main__':
    # test_inference()
    test_training()
    # test_loss()

    # load_pretrained_maskrcnn()
