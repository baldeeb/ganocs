from typing import List, Optional, Tuple, Dict

import torch 
from torch import nn, Tensor
from torchvision.models.detection.roi_heads import (
    RoIHeads, 
    fastrcnn_loss,
    maskrcnn_loss, 
    maskrcnn_inference,
    keypointrcnn_loss, keypointrcnn_inference,
    )
# from torchvision.models.detection.transform import paste_masks_in_image
from models.losses.nocs_loss import nocs_loss
from models.losses.multiview_consistency import multiview_consistency_loss
from models.nocs_util import select_nocs_proposals, separate_image_results
from models.nocs_heads import NocsHeads

import torchvision.transforms as T

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
        mask_roi_pool=None, mask_head=None, mask_predictor=None, 
        # Keypoint
        keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None,
        # NOCS
        in_channels:int     = 256, 
        num_bins:int        = 32,  # set to 1 for regression
        num_classes:int     = 91, 
        nocs_layers:List[int] = (256, 256, 256, 256, 256),
        cache_results:bool  = False,  # retains results at training
        nocs_loss=torch.nn.functional.cross_entropy,  # can be cross entropy or discriminator
        nocs_loss_mode:str  = 'classification',  # regression, classification
        multiheaded_nocs:bool = True,
        # Others
        **kwargs,
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
            # Keypoint
            keypoint_roi_pool, keypoint_head, keypoint_predictor,
        )

        self.cache_results, self.cache = cache_results, None
        self.nocs_loss_mode = nocs_loss_mode

        self.nocs_heads = NocsHeads(in_channels, nocs_layers,
                                    num_classes, num_bins,
                                    keys=['x', 'y', 'z'],
                                    multiheaded=multiheaded_nocs,
                                    mode=nocs_loss_mode)
        self.ignore_nocs = False
        self.nocs_loss = nocs_loss
        self._kwargs = kwargs
        self._training_mode = None
            
    def has_nocs(self): 
        if self.ignore_nocs: return False
        return self.nocs_heads is not None

    def _properly_allocate(self, x):
        '''This function is to be called on results when not training or
        when training and caching.'''
        if not self.training:
            return x
        elif self.cache_results: 
            if isinstance(x, dict):
                return {k:self._properly_allocate(v) for k,v in x.items()}
            if isinstance(x, list): 
                return [self._properly_allocate(xi) for xi in x]
            else: return x.clone().detach().cpu()
        else:
            raise RuntimeError('Missuse of function.')


    def _check_target_contents(self, targets):
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
                    
    def forward(self,
                features     : Dict[str, Tensor],
                proposals    : List[Tensor],
                image_shapes : List[Tuple[int, int]],
                targets      : Optional[List[Dict[str, Tensor]]] = None,
        ):
        """
        Args:
            features (List[Tensor]): features produced by the backbone 
            proposals (List[Tensor[N, 4]]): list of boxes regions.
            image_shapes (List[Tuple[H, W]]): the sizes of each image in the batch.
            targets (List[Dict]): 
        """
        if self._training_mode == 'multiview':
            return self.multiview_consistency(features, proposals, image_shapes, targets)
        
        if targets is not None: self._check_target_contents(targets)

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
            if labels is None: raise ValueError("labels cannot be None")
            if regression_targets is None: raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, 
                                                          labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        if not self.training or self.cache_results:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes":  self._properly_allocate(boxes[i]),
                        "labels": self._properly_allocate(labels[i]),
                        "scores": self._properly_allocate(scores[i]),
                    }
                )

        if self.has_mask():
            if self.training:
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                proposed_box_regions = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    proposed_box_regions.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

                if self.cache_results:
                    v = self._properly_allocate(pos_matched_idxs)
                    for i, vi in enumerate(v): result[i]['pos_matched_idxs'] = vi
            else:
                proposed_box_regions = [p["boxes"] for p in result]
                pos_matched_idxs = None

            mask_features = self.mask_roi_pool(features, proposed_box_regions, image_shapes)

            if self.has_nocs():
                nocs_proposals = self.nocs_heads(mask_features)

            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, proposed_box_regions, 
                                               gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}

                # Add NOCS to Loss
                if self.has_nocs():
                    gt_nocs = [t["nocs"] for t in targets]

                    reduction = 'none' if self.cache_results else 'mean'
                    loss_mask["loss_nocs"] = nocs_loss(gt_labels, gt_nocs, 
                                                       nocs_proposals, 
                                                       proposed_box_regions,
                                                       pos_matched_idxs,
                                                       reduction=reduction,
                                                       loss_fx=self.nocs_loss,
                                                       mode=self.nocs_loss_mode,
                                                       **self._kwargs)
                    
                    if self.cache_results:
                        split_loss = separate_image_results(loss_mask['loss_nocs'], labels)
                        split_nocs = separate_image_results(nocs_proposals, labels)
                        for i in range(len(result)):
                            result[i]['nocs'] = {k:self._properly_allocate(v[i]) 
                                                 for k, v in split_nocs.items()}
                            obj_loss = split_loss[i].mean((1,2,3))
                            result[i]['loss_nocs'] = self._properly_allocate(obj_loss)
                        loss_mask["loss_nocs"] = torch.mean(loss_mask["loss_nocs"])

            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob
                
                # Add NOCS to results
                if self.has_nocs():
                    nocs_maps = select_nocs_proposals(nocs_proposals, labels, 
                                                      self.nocs_heads.num_classes)
                    for n, r in zip(nocs_maps, result): r['nocs'] = n

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None):
            self._run_keypoint_head(result, targets, losses, proposals, matched_idxs, labels, features, image_shapes)

        if self.cache_results and self.training: self.cache = result
        return result, losses
    

    def training_mode(self, mode: str=None):
        assert mode in ['normal', 'multiview', None], 'unrecognized roi head training mode...'
        self._training_mode = mode


    def multiview_consistency(self,
                features     : Dict[str, Tensor],
                proposals    : List[Tensor],
                image_shapes : List[Tuple[int, int]],
                targets      : List[Dict[str, Tensor]],):
        """
        Args:
            features (List[Tensor]): features produced by the backbone 
            proposals (List[Tensor[N, 4]]): list of boxes regions.
            image_shapes (List[Tuple[H, W]]): the sizes of each image in the batch.
            depths (List[Tensor[N, H, W]]): list of depth maps.
            poses (List[Tensor[N, 4, 4]]): list of camera poses. Those are expected to be relative
                to a global reference frame.
        """
        assert self.has_mask() and self.has_nocs(), 'This function currently requires a mask head'
        results: List[Dict[str, torch.Tensor]] = []

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        boxes, scores, labels = self.postprocess_detections(class_logits, 
                                                            box_regression, 
                                                            proposals, 
                                                            image_shapes)
        for i in range(len(boxes)):
            results.append({"boxes":boxes[i], 
                            "labels":labels[i], 
                            "scores":scores[i],})

        proposed_box_regions = [p["boxes"] for p in results]
        mask_features = self.mask_roi_pool(features, proposed_box_regions, image_shapes)
        mask_head_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_head_features)
        labels = [r["labels"] for r in results]
        masks_probs = maskrcnn_inference(mask_logits, labels)
        for mask_prob, result in zip(masks_probs, results):
            result["masks"] = mask_prob
        
        # Add NOCS to results
        nocs_proposals = self.nocs_heads(mask_features)
        nocs_maps = select_nocs_proposals(nocs_proposals, labels, 
                                            self.nocs_heads.num_classes)
        for n, result in zip(nocs_maps, results): result['nocs'] = n

        return [], {'multiview_consistency_loss': multiview_consistency_loss(results, 
                                                                             targets, 
                                                                             scores, 
                                                                             image_shapes)}

    def _run_keypoint_head(self, result, targets, losses, proposals, matched_idxs, labels, features, image_shapes):
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


    @staticmethod
    def from_torchvision_roiheads(heads:RoIHeads, 
                                  nocs_num_bins=32, 
                                  **kwargs):
        '''Returns RoIHeadsWithNocs given an RoIHeads instance.'''
        nocs_ch_in = heads.mask_head[0][0].in_channels
        num_classes = heads.mask_predictor.mask_fcn_logits.out_channels
        return RoIHeadsWithNocs(
            heads.box_roi_pool, heads.box_head, heads.box_predictor,
            heads.proposal_matcher.high_threshold, heads.proposal_matcher.low_threshold,
            heads.fg_bg_sampler.batch_size_per_image, heads.fg_bg_sampler.positive_fraction,
            heads.box_coder.weights, 
            heads.score_thresh, heads.nms_thresh, heads.detections_per_img,
            heads.mask_roi_pool, heads.mask_head, heads.mask_predictor, 
            heads.keypoint_roi_pool, heads.keypoint_head, heads.keypoint_predictor,
            in_channels=nocs_ch_in, num_bins=nocs_num_bins, num_classes=num_classes,
            **kwargs
        )
