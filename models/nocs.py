from models.nocs_roi_heads import RoIHeadsWithNocs
from models.rcnn_transforms import GeneralizedRCNNTransformWithNocs
from torchvision.models.detection.mask_rcnn import MaskRCNN
import torch 

from typing import Any, Optional
from torch import nn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import (_validate_trainable_layers, 
                                                         _resnet_fpn_extractor)
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models._utils import _ovewrite_value_param
from torchvision.ops import misc as misc_nn_ops


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
        # NOCS parameters

        # nocs_bins=32,
        # cache_results=False,
        # nocs_loss=torch.nn.functional.cross_entropy,  # can be cross entropy or discriminator
        # multiheaded_nocs=False,
        # Others
        **kwargs,
    ):
        '''
        Args:
            cache_results (bool): indicates whether, at training time, the model
                should keep track of results that would be returned in eval mode.'''
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
        self.roi_heads = RoIHeadsWithNocs.from_torchvision_roiheads(
                                                    self.roi_heads,
                                                    # nocs_num_bins=nocs_bins,
                                                    # cache_results = cache_results,
                                                    # nocs_loss=nocs_loss,
                                                    **kwargs)
        
        self.cache = None
        
        # Update Transforms to include NOCS
        if image_mean is None: image_mean = [0.485, 0.456, 0.406]
        if image_std is None: image_std = [0.229, 0.224, 0.225]
        self._min_size, self._max_size = min_size, max_size
        self.transform = GeneralizedRCNNTransformWithNocs(
                                            min_size=min_size, 
                                            max_size=max_size, 
                                            image_mean=image_mean, 
                                            image_std=image_std, 
                                            **kwargs)
    
    def _updated_sizes(self, s):
        if isinstance(s, list): 
            return [self._updated_sizes(si) for si in s]
        min_size = torch.min(s).to(dtype=torch.float32)
        max_size = torch.max(s).to(dtype=torch.float32)
        scale = torch.min(self._min_size / min_size, 
                          self._max_size / max_size)
        return (s * scale).long()
        
    def forward(self, images, targets=None):
        result = super().forward(images, targets)
        if self.roi_heads.cache_results and self.training:
            original_sizes = torch.FloatTensor([img.shape[-2:] for img in images])
            new_size = self._updated_sizes(original_sizes)
            self.cache = self.transform.full_postprocess(
                                            self.roi_heads.cache, 
                                            new_size,
                                            original_sizes.long(),)
        return result


def get_nocs_resnet50_fpn(
    *,
    maskrcnn_weights: Optional[MaskRCNN_ResNet50_FPN_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
) -> NOCS:
    maskrcnn_weights = MaskRCNN_ResNet50_FPN_Weights.verify(maskrcnn_weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if maskrcnn_weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, 
                                            len(maskrcnn_weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = maskrcnn_weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = NOCS(backbone, num_classes=num_classes, **kwargs)

    if maskrcnn_weights is not None:
        model.load_state_dict(maskrcnn_weights.get_state_dict(progress=progress), 
                              strict=False)
        if maskrcnn_weights == MaskRCNN_ResNet50_FPN_Weights.COCO_V1:
            overwrite_eps(model, 0.0)

    return model