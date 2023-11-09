'''NOTE: This file is heavily based off of the ROIHeads class from torchvision.

Things to explain: 
    - Cache results: When experimenting with multiview consistency, it is useful
        to cache the results (not just losses) in the forward pass. This allows 
        us to calculate the consistency loss post forward pass.
    - 'no_nocs': This is a flag that can be set in the targets. If set, the
        nocs loss is not calculated for that object. This is useful when
        the data used does not have information to supervise nocs but is useful
        for other parts (mask, box, etc).
'''

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
from torchvision.models.detection.transform import paste_masks_in_image
from models.losses.nocs_loss import nocs_loss
from models.losses.multiview_consistency import multiview_consistency_loss
from models.nocs_util import select_nocs_proposals, separate_image_results
from models.nocs_heads import NocsHeads
from torch import (Tensor, cat)
import torch.nn.functional as F
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
        cache_results:bool  = False,  # retains results at training
        nocs_heads = None,
        
        # TODO: These should be owned by the NOCS head.
        nocs_loss_mode:str  = 'classification',  # regression, classification
        nocs_loss=torch.nn.functional.cross_entropy,  # can be cross entropy or discriminator

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

        # TODO: remove
        self.cache_results, self._cache = cache_results, None
        # TODO: remove pass through "**kwargs" instead
        self.nocs_loss_mode = nocs_loss_mode
        self.ignore_nocs = False
        self.nocs_loss = nocs_loss
        
        self.nocs_heads = nocs_heads
        self._kwargs = kwargs
        self._training_mode = None
        self._multiview_loss_mode = None
            
    def has_nocs(self): 
        if self.ignore_nocs: return False
        return self.nocs_heads is not None

    def _properly_allocate(self, x):
        '''This function is to be called on results when not training or
        when training and caching.'''
        if not self.training:
            return x
        elif self.cache_results or self.ignore_all_except_nocs: 
            if isinstance(x, dict):
                return {k:self._properly_allocate(v) for k,v in x.items()}
            if isinstance(x, list): 
                return [self._properly_allocate(xi) for xi in x]
            else: return x.clone().detach().cpu()
        else:
            raise RuntimeError('Missuse of function.')
        
    def process_without_gt(self, features, proposals, image_shapes,targets,result):
        """
        Computes boxes, scores, and labels for the given features, proposals, and image shapes
        without ground truth.
        
        Args:
            features_no_gt (List[Tensor]): features produced by the backbone without ground truth
            proposals_no_gt (List[Tensor[N, 4]]): list of box regions without ground truth.
            image_shapes_no_gt (List[Tuple[H, W]]): sizes of each image in the batch without ground truth.
            
        Returns:
            Tuple[List[Tensor], List[Tensor], List[Tensor]]: Returns boxes, scores, and labels respectively.
        """
        if self._training_mode == 'multiview':
            return self.multiview_consistency(features, proposals, image_shapes, targets)
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        boxes, scores, labels_no_gt = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(proposals)
        if not self.training or self.cache_results:
            for i in range(num_images):
                result.append(
                    {
                        "boxes":  self._properly_allocate(boxes[i]),
                        "labels": self._properly_allocate(labels_no_gt[i]),
                        "scores": self._properly_allocate(scores[i]),
                    }
                )
        proposed_box_regions = []
        pos_matched_idxs = []
        for img_id in range(num_images):
            pos = torch.where(labels_no_gt[img_id] > 0)[0]
            proposed_box_regions.append(proposals[img_id][pos])
        pos_matched_idxs = None
        mask_features_no_gt = self.mask_roi_pool(features, proposed_box_regions, image_shapes)
        mask_logits_no_gt = self.mask_predictor(mask_features_no_gt)
        if self.has_nocs():
            nocs_proposals_no_gt = self.nocs_heads(mask_features_no_gt)

        if len(targets)> 0 and 'depth' in targets[0]:
            depth = [t['depth'] for t in targets]

        nocs_gt_available = [not t.get('no_nocs', False) for t in targets]
        use_unlabeled_nocs = self._kwargs.get('use_unlabeled_nocs', False)
        
        return labels_no_gt,nocs_proposals_no_gt,proposed_box_regions,depth,nocs_gt_available,mask_logits_no_gt


    def _check_target_contents(self, targets):
        for i,t in enumerate(targets):
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            if not self.is_realsense[i]:
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
            features (Dict[str, Tensor]): features produced by the backbone 
            proposals (List[Tensor[N, 4]]): list of boxes regions.
            image_shapes (List[Tuple[H, W]]): the sizes of each image in the batch.
            targets (List[Dict]): 
        """



        self.is_realsense = [target['labels'] is not None for target in (targets or [])]

        proposals_with_gt = [proposal for proposal, is_real in zip(proposals, self.is_realsense) if is_real]
        proposals_no_gt = [proposal for proposal, is_real in zip(proposals, self.is_realsense) if not is_real]

        image_shapes_with_gt = [shape for shape, is_real in zip(image_shapes, self.is_realsense) if is_real]
        image_shapes_no_gt = [shape for shape, is_real in zip(image_shapes, self.is_realsense) if not is_real]
        if self.training:
            targets_with_gt = [target for target, is_real in zip(targets, self.is_realsense) if is_real]
            targets_no_gt = [target for target, is_real in zip(targets, self.is_realsense) if not is_real]
                   
        if self._training_mode == 'multiview':
            return self.multiview_consistency(features, proposals_with_gt, image_shapes_with_gt, targets_with_gt)
        
        #if targets is not None: self._check_target_contents(targets)
        gt_labels,proposed_box_regions,depth,nocs_gt_available,nocs_proposals,gt_nocs,gt_masks,pos_matched_idxs,loss_mask,losses,mask_logits =[],[],[],[],{},None,None,None,{},{},None
        result: List[Dict[str, torch.Tensor]] = []
        if proposals_with_gt:
            if self.training:
                proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals_with_gt, targets_with_gt)
            else:
                labels = None
                regression_targets = None
                matched_idxs = None

            box_features = self.box_roi_pool(features, proposals, image_shapes_with_gt)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)

            
            

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

            mask_features = self.mask_roi_pool(features, proposed_box_regions, image_shapes_with_gt)

            if self.has_nocs():
                nocs_proposals = self.nocs_heads(mask_features)

            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets_with_gt]
                gt_labels = [t["labels"] for t in targets_with_gt]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, proposed_box_regions, 
                                                gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
        if proposals_no_gt:
            labels_no_gt,nocs_proposals_no_gt,proposed_box_regions_no_gt,depth_no_gt,nocs_gt_available_no_gt,mask_logits_no_gt= self.process_without_gt(features,proposals_no_gt,image_shapes_no_gt,targets_no_gt,result)
            gt_labels=gt_labels+labels_no_gt
            proposed_box_regions=proposed_box_regions+proposed_box_regions_no_gt
            depth=depth+depth_no_gt
            nocs_gt_available=nocs_gt_available+nocs_gt_available_no_gt
            if not nocs_proposals:
                nocs_proposals=nocs_proposals_no_gt
            else:
                nocs_proposals = {key: torch.cat([nocs_proposals.get(key, torch.zeros([0, 7, 1, 28, 28])), nocs_proposals_no_gt.get(key, torch.zeros([0, 7, 1, 28, 28]))], dim=0) for key in set(nocs_proposals) | set(nocs_proposals_no_gt)}
        # Add NOCS to Loss ##########################################################################################
        if self.has_nocs() and self.training:
            gt_nocs = [t["nocs"].to(mask_logits.device) for t in targets_with_gt]

            if len(targets_with_gt)> 0 and 'depth' in targets_with_gt[0]:
                depth = [t['depth'] for t in targets_with_gt]

            nocs_gt_available = [not t.get('no_nocs', False) for t in targets_with_gt]
            use_unlabeled_nocs = self._kwargs.get('use_unlabeled_nocs', False)




            if any(nocs_gt_available) or use_unlabeled_nocs:

                reduction = 'none' if self.cache_results else 'mean'

                loss_mask["loss_nocs"] = nocs_loss(gt_labels, 
                                                    gt_nocs, 
                                                    gt_masks,
                                                    nocs_proposals, 
                                                    proposed_box_regions,
                                                    pos_matched_idxs,
                                                    reduction=reduction,
                                                    loss_fx=self.nocs_loss,
                                                    nocs_loss_mode=self.nocs_loss_mode,
                                                    depth=depth,
                                                    samples_with_valid_targets=nocs_gt_available,
                                                    **self._kwargs)
                
            if not all(nocs_gt_available) and self._kwargs["MIC_ON"]: 
                nocs_maps = select_nocs_proposals(nocs_proposals, gt_labels, 
                                                    self.nocs_heads.num_classes)
                nocs_out=[]
                bin_index=1
                for i, (pred, o_im_s) in enumerate(zip(nocs_maps, image_shapes)):

                    num_bins =  pred["x"].shape[bin_index]
                    def process_nocs_dim(v):
                        if num_bins > 1:  # Assume that it is classification and not regression.
                            v = v.argmax(bin_index) / num_bins 
                        v = paste_masks_in_image(v.unsqueeze(bin_index), 
                                                targets[i]["boxes"], o_im_s)
                        return v
                    nocs = cat([process_nocs_dim(pred["x"]),
                                process_nocs_dim(pred["y"]),
                                process_nocs_dim(pred["z"])],
                                dim=1)
                    nocs_maps[i]=nocs
                        

                reduction = 'none' if self.cache_results else 'mean'
                loss_mask["loss_MIC"] = F.mse_loss(nocs_maps[0], nocs_maps[1])
                #mic_loss(nocs_maps[0],nocs_maps[1])
                
                


                if self.cache_results:
                    split_loss = separate_image_results(loss_mask['loss_nocs'], labels)
                    split_nocs = separate_image_results(nocs_proposals, labels)
                    for i in range(len(result)):
                        result[i]['nocs'] = {k:self._properly_allocate(v[i]) 
                                            for k, v in split_nocs.items()}
                        obj_loss = split_loss[i].mean((1,2,3))
                        result[i]['loss_nocs'] = self._properly_allocate(obj_loss)
                    loss_mask["loss_nocs"] = torch.mean(loss_mask["loss_nocs"])
            
            #######################################################################################################################




            

        else:
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)
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
            labels = [r["labels"] for r in result]
            proposed_box_regions = [p["boxes"] for p in result]
            mask_features = self.mask_roi_pool(features, proposed_box_regions, image_shapes)

            if self.has_nocs():
                nocs_proposals = self.nocs_heads(mask_features)

            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)
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

        if self.cache_results and self.training: self._cache = result

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
                                                                             image_shapes,
                                                                             mode=self._multiview_loss_mode)}

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
                                  **kwargs):
        '''Returns RoIHeadsWithNocs given an RoIHeads instance.'''
        nocs_kwargs = {
            'in_channels':heads.mask_head[0][0].in_channels,
            'num_classes':heads.mask_predictor.mask_fcn_logits.out_channels
        }
        keys_map = {'nocs_layers'       :'layers',
                    'nocs_num_bins'     :'num_bins',
                    'multiheaded_nocs'  :'multiheaded',
                    'nocs_keys'         :'keys',
                    'nocs_loss_mode'    :'mode',}
        for k,v in keys_map.items():
            if k in kwargs: nocs_kwargs[v]=kwargs[k]
        nocs_heads = NocsHeads(**nocs_kwargs)

        return RoIHeadsWithNocs(
            heads.box_roi_pool, heads.box_head, heads.box_predictor,
            heads.proposal_matcher.high_threshold, heads.proposal_matcher.low_threshold,
            heads.fg_bg_sampler.batch_size_per_image, heads.fg_bg_sampler.positive_fraction,
            heads.box_coder.weights, 
            heads.score_thresh, heads.nms_thresh, heads.detections_per_img,
            heads.mask_roi_pool, heads.mask_head, heads.mask_predictor, 
            heads.keypoint_roi_pool, heads.keypoint_head, heads.keypoint_predictor,
            nocs_heads=nocs_heads,
            **kwargs
        )
