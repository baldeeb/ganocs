import torch 
from torchvision.ops import boxes as _, roi_align
from torch.nn.functional import (cross_entropy, 
                                 binary_cross_entropy, 
                                 softmax)
from models.nocs_util import select_labels_in_dict
from models.discriminators import (DiscriminatorWithOptimizer, 
                                  MultiDiscriminatorWithOptimizer,
                                  MultiClassDiscriminatorWithOptimizer, 
                                  DepthAwareDiscriminator,
                                  RgbdMultiDiscriminatorWithOptimizer)
from typing import Union, Dict, Callable
from models.losses.symmetry_aware_loss import SymmetryAwareLoss

def project_on_boxes(data, boxes, matched_idxs, M)->torch.Tensor:
    """ Code borrowed from torchvision.models.detection.roi_heads.project_masks_on_boxes
    Given data, returns RoIAligned data by cropping the boxes and resizing them to MxM squares.
    Args:
        data (Tensor): [C, H, W] Each element contains ``C`` feature maps 
            of dimensions ``H x W``.
        boxes (Tensor): [N, 4] The box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
        matched_idxs (Tensor): [N] Index of the box pertaining 
    Returns (Tensor): [C, N, M, M] containin list of resized data.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    return roi_align(data.transpose(0, 1), rois, (M, M), 1.0)

def nocs_preprocessing_for_discriminator(
                          proposals:torch.Tensor, 
                          mode='classification'):
    
    if mode == 'classification':
        # Get weighted average of proposal
        N = proposals.shape[2]
        bin_idxs = torch.arange(N).to(proposals)
        bin_idxs = bin_idxs[None, None, :, None, None]
        soft_nocs = softmax(proposals, dim=2)
        prediction = (soft_nocs * bin_idxs).sum(dim=2) / N
    elif mode == 'regression':
        assert proposals.shape[2] == 1, 'Regression only supports 1 bin'
        prediction = proposals.squeeze(2)
    else:
        raise ValueError(f'Unknown mode {mode}')
    return prediction

def discriminator_as_loss(discriminator:Union[DiscriminatorWithOptimizer,
                                              MultiDiscriminatorWithOptimizer,
                                              MultiClassDiscriminatorWithOptimizer,
                                              DepthAwareDiscriminator], 
                          proposals:torch.Tensor, 
                          targets:torch.Tensor,
                          reduction:str='mean',
                          mode='classification',
                          has_gt=None,
                          classes:torch.Tensor=None,
                          depth:torch.Tensor=None,
                          disc_steps:int=1):
    '''
    Args:
        proposals (Tensor): [B, 3, N, H, W] tensor of predicted nocs
            where B is batch size, 3 is for x,y,z channels, N is the
            number of bins, and H, W are the height and width of the
            image.
        targets (Tensor): [B, 3, H, W] tensor of indices indicating
            which of the binary logits is the correct nocs value.
        classes (Tensor): [B] constains the class of the sample. Used
            when using a different discriminator for each class.
    '''

    update_real_kwargs, update_fake_kwargs, forward_kwargs = {}, {}, {}
    prediction = nocs_preprocessing_for_discriminator(proposals, mode)
    select_with_gt = lambda x : x if has_gt is None else x[has_gt]

    # Select targets used to train the discriminator
    selected_targets = select_with_gt(targets)
    
    # Quick test: For concatenating depth to NOCS
    if isinstance(discriminator, RgbdMultiDiscriminatorWithOptimizer):
        assert depth is not None, 'Depth is needed for depth aware discriminators.'
        prediction = torch.cat([prediction, depth], dim=1)
        selected_targets = torch.cat([selected_targets, select_with_gt(depth)], dim=1)

    if classes is not None:
        # For multihead discriminator, set classes that will specify head
        update_real_kwargs['classes'] = select_with_gt(classes)
        update_fake_kwargs['classes'] = classes
        forward_kwargs['classes']     = classes 

    # For contextual discriminator
    # if isinstance(discriminator, DepthAwareDiscriminator):
    if 'depth_context' in discriminator.properties:
        # assert depth is not None, 'Depth is needed for depth aware discriminators.'
        update_real_kwargs['ctx'] = select_with_gt(depth)
        update_fake_kwargs['ctx'] = depth
        forward_kwargs['ctx']     = depth 
    
    # Train discriminator
    for _ in range(disc_steps):
        discriminator.update(real_data=selected_targets,
                            fake_data=prediction,
                            real_kwargs=update_real_kwargs,
                            fake_kwargs=update_fake_kwargs)
    # Get nocs loss
    return discriminator.critique(x=prediction, **forward_kwargs)


def get_list_samples_with_gt_nocs(x, nocs_gt_available):
    return [vi for vi, s in zip(x, nocs_gt_available) if s]

def get_indices_of_detections_with_gt_nocs(nocs_gt_available, matched_idxs):
    len_s = [len(v) for v in matched_idxs]
    zs, os = lambda a: torch.zeros(a, dtype=torch.bool), lambda a: torch.ones(a, dtype=torch.bool)
    return torch.cat([os(l) if i else zs(l) for i, l in zip(nocs_gt_available, len_s)])

def get_dict_samples_with_gt_nocs(x, detections_with_gt_nocs):
    return {k:v[detections_with_gt_nocs] for (k, v) in x.items()}

def get_loss_objectives(loss_fx):
    cross_entropy_loss, discriminator_loss, mse_loss = None, None,None
    if isinstance(loss_fx, torch.nn.ModuleDict):
        discriminator_loss = loss_fx['discriminator'] if 'discriminator' in loss_fx else None
        cross_entropy_loss = loss_fx['cross_entropy'] if 'cross_entropy' in loss_fx else None
        mse_loss = loss_fx['mse'] if 'mse' in loss_fx else None
    elif isinstance(loss_fx, torch.nn.CrossEntropyLoss):
        cross_entropy_loss = loss_fx
    elif isinstance(loss_fx, torch.nn.MSELoss):
        mse_loss = loss_fx
    else: 
        discriminator_loss = loss_fx
    # if discriminator_loss is not None:
    #     assert isinstance(discriminator_loss, 
    #                         (DiscriminatorWithOptimizer, 
    #                         MultiDiscriminatorWithOptimizer,
    #                         MultiClassDiscriminatorWithOptimizer,
    #                         DepthAwareDiscriminator))
    return cross_entropy_loss, mse_loss, discriminator_loss

def get_loss_weights(loss_weights, default=lambda:1.0):
    return {k:loss_weights.get(k, default) for k in ['cross_entropy', 'mse', 'discriminator']}

def nocs_loss(gt_labels, 
              gt_nocs,
              gt_masks,
              nocs_proposals, 
              box_proposals, 
              matched_ids, 
              reduction='mean',
              loss_fx:Union[Dict, Callable]=cross_entropy,
              nocs_loss_mode='classification', # regression or classification
              dispersion_loss=None, # TODO: integrate into loss_fx
              dispersion_weight=0.0, # TODO: substitute with loss_weights
              depth=None,
              samples_with_valid_targets=None,
              **kwargs):
    '''
    Calculates nocs loss. Supports cross_entropy and discriminator loss.
    Args: 
        gt_labels (List[long]): the labels existant in each image 
            in the batch. The length of this list is the batch size.
        gt_nocs [B, 3, H, W] (float): ground truth nocs with
            values in [0, 1]
        nocs_proposals Dict[str, Tensor]: A dictionary of (x,y,z) of
                tensors containing the predicted nocs maps [B, C, N, H, W]
                Where C is the number of classes and N is bins.
        proposed_box_regions List[Tensor]: Where each element of the list
            is a tensor of shape [N, 4] containing the bounding box of the
            region of interest in an image.
        matched_idxs (List[Tensor]): A [B, N] set of indices indicating the
            matching ground truth of each proposal.
        reduction (str): 'none', 'mean' or other pytorch's cross entropy 
            reductions.
        dispersion_loss (Callable): A function that takes the nocs proposals
            and returns a scalar loss per batch.
        samples_with_valid_targets (list[bool]): A list of booleans indicating,
            for each image in the batch, whether or not it has sufficient 
            information to superervise nocs. When useing l2 loss this means
            having gt nocs. If None, it will be calculated from the gt_nocs.

    Returns 
        loss (Tensor): An element or list of values depending on the reduction
    
    NOTE: Symmetry loss is not implemented
    '''

    # Find samples with gt nocs
    if samples_with_valid_targets is None:
        samples_with_valid_targets = [g.sum().item() > 0 for g in gt_nocs]
    
    detections_with_gt_nocs = get_indices_of_detections_with_gt_nocs(
                                                        samples_with_valid_targets,
                                                        matched_ids),
    select_with_gt_nocs = lambda x : x[detections_with_gt_nocs]
    
    entropy_loss, mse_loss, discriminator_loss = get_loss_objectives(loss_fx)
    loss_weight_fxs = get_loss_weights(kwargs.get('loss_weights', {}))

    # Select the label for each proposal
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, matched_ids)]
    proposals = select_labels_in_dict(nocs_proposals, labels)  # Dict (3 values) [T, N, H, W] 
    proposals = torch.stack(tuple(proposals.values()), dim=1) # [T, 3, N, H, W] 
    device = proposals.device
    masked_nocs = [n[:, None] * m[None].to(device) for n, m in zip(gt_nocs, gt_masks)]

    # Reshape to fit box by performing roi_align
    W = proposals.shape[-1]  # Width of proposal we want gt to match
    targets = [project_on_boxes(m, p, i, W) for m, p, i in zip(masked_nocs, box_proposals, matched_ids)]
    targets = torch.cat(targets, dim=0) # [B, 3, H, W]
    
    loss = torch.tensor(0.0).to(device)
    if entropy_loss:
        # If target is empty return 0
        if targets.numel() == 0: return proposals.sum() * 0
        assert nocs_loss_mode == 'classification', 'Cross entropy only supports classification'
        loss_kwargs = {}
        if isinstance(entropy_loss, SymmetryAwareLoss):
            loss_kwargs['labels'] = select_with_gt_nocs(torch.cat(labels))
            loss_kwargs['reduction']=reduction
            loss_kwargs['loss_type']='cross_entropy'
            loss += entropy_loss(select_with_gt_nocs(proposals), 
                        select_with_gt_nocs(targets), 
                        **loss_kwargs,
                            ) * loss_weight_fxs['cross_entropy']()
        else:
            targets_discretized = (targets * (proposals.shape[2] - 1)).round().long()  # (0->1) to indices [0, 1, ...]
            loss += cross_entropy(select_with_gt_nocs(proposals).transpose(1,2), 
                                select_with_gt_nocs(targets_discretized), 
                                reduction=reduction
                    ) * loss_weight_fxs['cross_entropy']()
        if False:
            # Temperature to limit the proposal probabilities.
            thresh = 1e4
            pmin, pmax = proposals.min(), proposals.max() 
            tau = min([thresh/abs(pmin), thresh/pmax, 1.0])
            proposals = proposals * tau # multiply by temperature

    if mse_loss is not None:
        assert proposals.shape[2] == 1, 'Expecting only single bin per color.'
        loss_kwargs = {}
        if isinstance(mse_loss, SymmetryAwareLoss):
            loss_kwargs['labels'] = select_with_gt_nocs(torch.cat(labels))
            loss_kwargs['loss_type']='mse'
        loss += mse_loss(
                select_with_gt_nocs(proposals).squeeze(2), 
                select_with_gt_nocs(targets), 
                **loss_kwargs,
            ) * loss_weight_fxs['mse']()

    if discriminator_loss is not None: 
        disc_kwargs = {
            'has_gt':     detections_with_gt_nocs,
            'classes':    torch.cat(labels) if 'multiclass' in discriminator_loss.properties else None,
            'depth':      None,
            'disc_steps': kwargs.get('discriminator_steps_per_batch', 1)
        }

        masks = torch.cat([project_on_boxes(m[None].to(p), p, i, W) 
                            for m, p, i in zip(gt_masks, box_proposals, matched_ids)])

        if True:  # 'depth_context' in discriminator_loss.properties:
            masked_depth = [(d[:, None].to(device) * m[None].to(device)) for d, m in zip(depth, gt_masks)]
            target_depths = [project_on_boxes(m, p, i, W) for m, p, i in zip(masked_depth, box_proposals, matched_ids)]
            target_depths = torch.cat(target_depths, dim=0) # [B, 1, H, W]
            
            mu = torch.stack([(d + ((1-m)*9999)).min() for d, m in zip(target_depths, masks)])  # min normalize
            # mu = disc_kwargs['depth'].mean((-2, -1))                                          # mean normalize
            # mu = disc_kwargs['depth'].flatten(-2, -1).median(-1).values                       # median normalize
            disc_kwargs['depth'] = target_depths - (masks * mu[:, None, None, None])
            
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # WHAT IF WE CONCAT MASK AND DEPTH -> better inform discriminator
        # proposals = torch.cat([proposals, masks[:, :, None], disc_kwargs['depth'][:, :, None]], dim=1)
        # targets = torch.cat([targets, masks, disc_kwargs['depth']], dim=1)
        # proposals = torch.cat([proposals, masks[:, :, None]], dim=1)
        # targets = torch.cat([targets, masks], dim=1)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


        loss += discriminator_as_loss(discriminator_loss, 
                                     proposals, 
                                     targets,
                                     reduction=reduction,
                                     mode=nocs_loss_mode,
                                     **disc_kwargs
                    ) * loss_weight_fxs['discriminator']()
    
    if dispersion_loss is not None:
        # Motivates the distribution of the proposal to be similar to that of target
        loss += dispersion_loss(proposals, targets) * dispersion_weight

    return loss