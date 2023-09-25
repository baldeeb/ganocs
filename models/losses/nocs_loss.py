import torch 
from torchvision.ops import boxes as _, roi_align
from torch.nn.functional import (cross_entropy, 
                                 binary_cross_entropy, 
                                 softmax)
from models.nocs_util import select_labels_in_dict
from models.discriminator import (DiscriminatorWithOptimizer, 
                                  MultiClassDiscriminatorWithOptimizer, 
                                  ContextAwareDiscriminator)

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

def discriminator_as_loss(discriminator:DiscriminatorWithOptimizer, 
                          proposals:torch.Tensor, 
                          targets:torch.Tensor,
                          reduction:str='mean',
                          mode='classification',
                          has_gt=None,):
    '''
    Args:
        proposals (Tensor): [B, 3, N, H, W] tensor of predicted nocs
            where B is batch size, 3 is for x,y,z channels, N is the
            number of bins, and H, W are the height and width of the
            image.
        targets (Tensor): [B, 3, H, W] tensor of indices indicating
            which of the binary logits is the correct nocs value.'''
    
    prediction = nocs_preprocessing_for_discriminator(proposals,
                                                      mode)
    # Train discriminator
    if has_gt is not None:
        discriminator.update(targets[has_gt], prediction[has_gt])
    else:
        discriminator.update(targets, prediction)
    
    # Get nocs loss
    l = discriminator(prediction)
    return binary_cross_entropy(l, torch.ones_like(l), reduction=reduction)

    
def multiclass_discriminator_as_loss(discriminator:DiscriminatorWithOptimizer, 
                          proposals:torch.Tensor, 
                          targets:torch.Tensor,
                          classes:torch.Tensor,
                          reduction:str='mean',
                          mode='classification',
                          has_gt=None,):
    '''
    Args:
        proposals (Tensor): [B, 3, N, H, W] tensor of predicted nocs
            where B is batch size, 3 is for x,y,z channels, N is the
            number of bins, and H, W are the height and width of the
            image.
        targets (Tensor): [B, 3, H, W] tensor of indices indicating
            which of the binary logits is the correct nocs value.
        classes (Tensor): [B] is a list of the class ids associated 
            with the B targets.'''
    
    prediction = nocs_preprocessing_for_discriminator(proposals,
                                                      mode)
    # Train discriminator
    if has_gt is not None:
        discriminator.update(targets[has_gt], prediction[has_gt], classes[has_gt])
    else:
        discriminator.update(targets, prediction, classes)
    
    # Get nocs loss
    l = discriminator(prediction, classes)
    return binary_cross_entropy(l, torch.ones_like(l), reduction=reduction)

    
def contextualized_discriminator_as_loss(discriminator:DiscriminatorWithOptimizer, 
                          proposals:torch.Tensor, 
                          targets:torch.Tensor,
                          classes:torch.Tensor,
                          depth:torch.Tensor,
                          reduction:str='mean',
                          mode='classification',):
    raise NotImplementedError('Contextualized discriminator is not implemented yet.')
    '''
    Args:
        proposals (Tensor): [B, 3, N, H, W] tensor of predicted nocs
            where B is batch size, 3 is for x,y,z channels, N is the
            number of bins, and H, W are the height and width of the
            image.
        targets (Tensor): [B, 3, H, W] tensor of indices indicating
            which of the binary logits is the correct nocs value.
        classes (Tensor): [B] is a list of the class ids associated 
            with the B targets.'''
    
    prediction = nocs_preprocessing_for_discriminator(proposals,
                                                      mode)

    context = None # TODO: encode classes, depth

    # Train discriminator
    discriminator.update(targets, prediction, context)
    
    # Get nocs loss
    l = discriminator(prediction, context)
    return binary_cross_entropy(l, torch.ones_like(l), reduction=reduction)


def get_list_samples_with_gt_nocs(x, nocs_gt_available):
    return [vi for vi, s in zip(x, nocs_gt_available) if s]

def get_indices_of_detections_with_gt_nocs(nocs_gt_available, matched_idxs):
    len_s = [len(v) for v in matched_idxs]
    zs, os = lambda a: torch.zeros(a, dtype=torch.bool), lambda a: torch.ones(a, dtype=torch.bool)
    return torch.cat([os(l) if i else zs(l) for i, l in zip(nocs_gt_available, len_s)])

def get_dict_samples_with_gt_nocs(x, nocs_gt_available, matched_idxs):
    mask = get_indices_of_detections_with_gt_nocs(nocs_gt_available, matched_idxs)
    return {k:v[mask] for (k, v) in x.items()}


def nocs_loss(gt_labels, 
              gt_nocs,
              gt_masks,
              nocs_proposals, 
              proposed_box_regions, 
              matched_ids, 
              reduction='mean',
              loss_fx=cross_entropy,
              mode='classification', # regression or classification
              dispersion_loss=None,
              dispersion_weight=0.0,
              depth=None,
              samples_with_valid_targets=None,
              **_):
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
    
    if loss_fx == cross_entropy:
        # Only keep samples with valid targets
        gt_labels = get_list_samples_with_gt_nocs(gt_labels, samples_with_valid_targets)
        gt_nocs = get_list_samples_with_gt_nocs(gt_nocs, samples_with_valid_targets)
        gt_masks = get_list_samples_with_gt_nocs(gt_masks, samples_with_valid_targets)
        nocs_proposals = get_dict_samples_with_gt_nocs(nocs_proposals, samples_with_valid_targets, matched_ids)
        proposed_box_regions = get_list_samples_with_gt_nocs(proposed_box_regions, samples_with_valid_targets)
        matched_ids = get_list_samples_with_gt_nocs(matched_ids, samples_with_valid_targets)
        depth = get_list_samples_with_gt_nocs(depth, samples_with_valid_targets)

    # Select the label for each proposal
    labels = [gt_label[idxs] for gt_label, idxs 
              in zip(gt_labels, matched_ids)]
    proposals = select_labels_in_dict(nocs_proposals, labels)  # Dict (3 values) [T, N, H, W] 
    proposals = torch.stack(tuple(proposals.values()), dim=1) # [T, 3, N, H, W] 
    masked_nocs = [n[:, None] * m[None].to(n) for n, m in zip(gt_nocs, gt_masks)]

    # Reshape to fit box by performing roi_align
    W = proposals.shape[-1]  # Width of proposal we want gt to match
    targets = [project_on_boxes(m, p, i, W) 
        for m, p, i in zip(masked_nocs, proposed_box_regions, matched_ids)]
    targets = torch.cat(targets, dim=0) # [B, 3, H, W]

    if loss_fx == cross_entropy:

        # If target is empty return 0
        if targets.numel() == 0: return proposals.sum() * 0
        
        assert mode == 'classification', 'Cross entropy only supports classification'
        targets_idxs = (targets * (proposals.shape[2] - 1)).round().long()  # (0->1) to indices [0, 1, ...]

        # Temperature to limit the proposal probabilities.
        if False:
            thresh = 1e4
            pmin, pmax = proposals.min(), proposals.max() 
            tau = min([thresh/abs(pmin), thresh/pmax, 1.0])
            proposals = proposals * tau # multiply by temperature
            # assert not proposals.isnan().any(), 'Proposals are NAN after temperature.'

        loss = cross_entropy(proposals.transpose(1,2), 
                             targets_idxs,
                             reduction=reduction)
    
    # TODO: make base class for all discriminator losses to inherit.
    elif isinstance(loss_fx, DiscriminatorWithOptimizer):
        loss = discriminator_as_loss(loss_fx, 
                                     proposals, 
                                     targets,
                                     reduction=reduction,
                                     mode=mode,
                                     has_gt=get_indices_of_detections_with_gt_nocs(
                                                        samples_with_valid_targets,
                                                        matched_ids),
                                    )
    elif isinstance(loss_fx, MultiClassDiscriminatorWithOptimizer):
        loss = multiclass_discriminator_as_loss(loss_fx, 
                                                proposals, 
                                                targets, 
                                                torch.cat(labels),
                                                reduction=reduction,
                                                mode=mode,
                                                has_gt=get_indices_of_detections_with_gt_nocs(
                                                                    samples_with_valid_targets,
                                                                    matched_ids),
                                                )
    elif isinstance(loss_fx, ContextAwareDiscriminator):
        depth_crop = [project_on_boxes(m, p, i, W) 
            for m, p, i in zip(depth, proposed_box_regions, matched_ids)]
        loss = contextualized_discriminator_as_loss(loss_fx, 
                                                proposals, 
                                                targets, 
                                                labels,
                                                reduction=reduction,
                                                mode=mode,
                                                depth=depth_crop)

    if dispersion_loss is not None:
        # Motivates the distribution of the proposal to be similar to that of target
        loss += dispersion_loss(proposals, targets) * dispersion_weight

    return loss