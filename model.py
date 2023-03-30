
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import torch 
from torch import nn, Tensor

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNHeads, MaskRCNNPredictor


class RoIHeadsWithNocs(nn.Module):
    def __init__(self, in_channels, num_bins=32, other_heads:RoIHeads=None):
        super().__init__()
        self.heads = other_heads

        layers = (256, 256, 256, 256, 256)

        self.nocs_heads = {}
        for k in ['x', 'y', 'z']:
            self.nocs_heads[k] = nn.ModuleDict({
                'roi_align': MultiScaleRoIAlign(
                                featmap_names=["0", "1", "2", "3"], 
                                output_size=14, 
                                sampling_ratio=2),
                'head': MaskRCNNHeads(
                                in_channels, 
                                layers[:-1], 
                                dilation=1),
                'pred': MaskRCNNPredictor(
                                layers[-2], 
                                layers[-1], 
                                num_bins)
            })

    def forward(
            self,
            features     : Dict[str, Tensor],
            proposals    : List[Tensor],
            image_shapes : List[Tuple[int, int]],
            targets      : Optional[List[Dict[str, Tensor]]] = None ,
            ):
        
        results, lables = self.heads(features, proposals, image_shapes, targets)
        print(results[0].keys())
        for key, head in self.nocs_heads.items():
            x = head['roi_align'](features, proposals, image_shapes)
            x = head['head'](x)
            x = head['pred'](x)



        return results, lables
    

def discretize_nocs(mast_gt_nocs):
# TODO ################################################
    pass

def nocs_map_loss(gt_mask, gt_nocs, pred_nocs):
# TODO ################################################
# NOTE: maybe instead of feeding in gt_nocs, we can 
#       feed in the discretized version of it.
    '''
    Takes in an instance segmentation mask along with
    the ground truth and predicted nocs maps. 

    Args: 
        gt_mask [B, C, H, W] (uint8): instance segmentation
        gt_nocs [B, 3, H, W] (uint8): ground truth nocs with 
                channel values 0-255.
        pred_nocs [B, 3, C, H, W] (bool): Discretized classification
                of the nocs map. C here indicates number of bins.
    
    returns [N]: A dictionary of lists containing loss values
    '''
    pass

class NOCS(MaskRCNN):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Mask parameters
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        **kwargs,
    ):
        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            # Mask parameters
            mask_roi_pool,
            mask_head,
            mask_predictor,
            **kwargs,
        )
        self.roi_heads = RoIHeadsWithNocs(in_channels=backbone.out_channels, other_heads=self.roi_heads)

    def forward(self, images, targets=None):
        mask_rcnn_results = MaskRCNN.forward(self, images, targets)
        return mask_rcnn_results



if __name__=='__main__':


    import torch
    import torchvision
    from pathlib import Path
    from torchvision import transforms
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone    

    if 'image' not in vars():
        # DATA_FOLDER = Path(__file__).resolve().parent / 'data'
        DATA_FOLDER = Path('/home/baldeeb/Data/cocodataset/')
        dataset = torchvision.datasets.CocoDetection(
                                    DATA_FOLDER/'val2017' 
                                    ,DATA_FOLDER/'annotations_trainval2017/annotations/instances_val2017.json'
                                    ,transform=transforms.PILToTensor())
        image, targets = dataset[0][0], dataset[0][1] 


    backbone = resnet_fpn_backbone('resnet50', ResNet50_Weights.DEFAULT)
    m = NOCS(backbone, 91)


    m.eval()
    x = [torch.rand(3, 30, 40), torch.rand(3, 50, 40)]
    # x = [image.float() / 255.0]
    predictions = m(x)