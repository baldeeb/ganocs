
import torch
import torchvision
from pathlib import Path
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone    

from models.nocs import NOCS

if __name__=='__main__':


    # if 'image' not in vars():
    #     ## Laptop
    #     DATA_FOLDER = Path(__file__).resolve().parent / 'data'
    #     IMG_FOLDER = Path('train')
    #     ANNOTATION_FOLDER = Path('annotations_trainval2017/annotations/instances_train2017.json')

    #     ## Lab PC
    #     # DATA_FOLDER = Path('/home/baldeeb/Data/cocodataset/')
    #     # IMG_FOLDER = Path('val2017')
    #     # ANNOTATION_FOLDER = Path('annotations_trainval2017/annotations/instances_val2017.json')

    #     dataset = torchvision.datasets.CocoDetection(
    #                                 DATA_FOLDER/IMG_FOLDER
    #                                 ,DATA_FOLDER/ANNOTATION_FOLDER
    #                                 ,transform=transforms.PILToTensor())
    #     image, targets = dataset[0][0], dataset[0][1] 

        



    backbone = resnet_fpn_backbone('resnet50', ResNet50_Weights.DEFAULT)
    m = NOCS(backbone, 2)

    m.eval()
    x = [torch.rand(3, 10, 10)]
    # x = [image.float() / 255.0]
    predictions = m(x)



def show_image(im):
    pass
