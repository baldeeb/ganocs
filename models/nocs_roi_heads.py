from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import torch 
from torch import nn, Tensor

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import (
    RoIHeads, fastrcnn_loss,
    maskrcnn_loss, maskrcnn_inference,
    keypointrcnn_loss, keypointrcnn_inference,
    )
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNHeads, MaskRCNNPredictor


def add_nocs_to_RoIHeads(heads:RoIHeads, nocs_ch_in=256, nocs_num_bins=32):
    return RoIHeadsWithNocs(
        heads.box_roi_pool, 
        heads.box_head,
        heads.box_predictor,

        heads.proposal_matcher.high_threshold,
        heads.proposal_matcher.low_threshold,

        heads.fg_bg_sampler.batch_size_per_image,
        heads.fg_bg_sampler.positive_fraction,
        
        heads.box_coder.weights, 

        heads.score_thresh,
        heads.nms_thresh, 
        heads.detections_per_img,

        heads.mask_roi_pool, 
        heads.mask_head,
        heads.mask_predictor, 

        heads.keypoint_roi_pool,
        heads.keypoint_head,
        heads.keypoint_predictor,

        in_channels=nocs_ch_in,
        num_bins=nocs_num_bins
    )


class RoIHeadsWithNocs(RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        # NOCS
        in_channels=256,
        num_bins=32,
    ):
        super().__init__(
            box_roi_pool, box_head, box_predictor,
            # Faster R-CNN training
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
            positive_fraction, bbox_reg_weights,
            # Faster R-CNN inference
            score_thresh, nms_thresh, detections_per_img,
            # Mask
            mask_roi_pool, mask_head, mask_predictor,
            keypoint_roi_pool, keypoint_head, keypoint_predictor,
        )

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
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)


        if self.nocs_heads:
            if not self.has_mask(): raise Exception("The mask head is needed for ")
            nocs_results = {}
            for key, layers in self.nocs_heads.items():
                # TODO: same as mask head
                x = layers['roi_align'](features, proposals, image_shapes)
                x = layers['head'](x)
                nocs_results[key] = layers['pred'](x)

            for r_idx, r in enumerate(result):
                r["nocs"] = {k:v[r_idx] for k, v in nocs_results.items()}


        return result, losses
    