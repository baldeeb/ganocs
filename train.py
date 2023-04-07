
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
    
    return dataset

if __name__=='__main__':

    # Initialize Model
    backbone = resnet_fpn_backbone('resnet50', ResNet50_Weights.DEFAULT)
    m = NOCS(backbone, 2)
    # m = MaskRCNN(backbone, 2)
    m.eval()

    dataset = get_data()

    # Run loop
    for images, targets in dataset:
        predictions = m([images])
        print('done')
        break


def show_image(im):
    pass
