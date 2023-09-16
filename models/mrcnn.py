# from collections import OrderedDict
from typing import Any, Callable, Optional

from torch import nn
# from torchvision.ops import MultiScaleRoIAlign

from torchvision.ops import misc as misc_nn_ops
# from torchvision.transforms._presets import ObjectDetection
# from torchvision.models._api import register_model, Weights, WeightsEnum
# from torchvision.models._meta import _COCO_CATEGORIES
# from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
# from torchvision.models.detection.faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead, 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN
from torchvision.models.detection.roi_heads import RoIHeads

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

def maskrcnn_resnet50_fpn_adjusted_class_count(
    *,
    weights: Optional[MaskRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> MaskRCNN:
    """ Mostly a copy of torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn """

    model = maskrcnn_resnet50_fpn(weights=weights,
                                  progress=progress,
                                  num_classes=len(weights.meta["categories"]),
                                  weights_backbone=weights_backbone,
                                  trainable_backbone_layers=trainable_backbone_layers,
                                  **kwargs)
        
    if num_classes != len(weights.meta["categories"]):

        # # Method 1: change last layer.
        # model.roi_heads.box_predictor.cls_score = nn.Linear(
        #     model.roi_heads.box_predictor.cls_score.in_features, num_classes
        # )
        # model.roi_heads.box_predictor.num_classes = num_classes
        # model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(
        #     model.roi_heads.mask_predictor.mask_fcn_logits.in_channels, num_classes, 1, 1, 0
        # )

        # Method 2: change all of the roi_heads
        heads = maskrcnn_resnet50_fpn(progress=progress,
                                    num_classes=num_classes,
                                    **kwargs).roi_heads
        model.roi_heads = heads

    return model