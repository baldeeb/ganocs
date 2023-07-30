import torch 
from torchvision.ops import boxes as _, roi_align
from torch.nn.functional import (cross_entropy, 
                                 binary_cross_entropy, 
                                 softmax)
from models.nocs_util import select_labels
from models.discriminator import DiscriminatorWithOptimizer, MultiClassDiscriminatorWithOptimizer

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
                          mode='classification'):
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
    discriminator.update(targets, prediction)
    
    # Get nocs loss
    l = discriminator(prediction)
    return binary_cross_entropy(l, torch.ones_like(l), reduction=reduction)

    
def multiclass_discriminator_as_loss(discriminator:DiscriminatorWithOptimizer, 
                          proposals:torch.Tensor, 
                          targets:torch.Tensor,
                          classes:torch.Tensor,
                          reduction:str='mean',
                          mode='classification'):
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
    discriminator.update(targets, prediction, classes)
    
    # Get nocs loss
    l = discriminator(prediction)
    return binary_cross_entropy(l, torch.ones_like(l), reduction=reduction)

    

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
    Returns 
        loss (Tensor): An element or list of values depending on the reduction
    
    NOTE: Symmetry loss is not implemented
    '''
    # Select the label for each proposal
    labels = [gt_label[idxs] for gt_label, idxs 
              in zip(gt_labels, matched_ids)]
    proposals = select_labels(nocs_proposals, labels)  # Dict (3 values) [T, N, H, W] 
    proposals = torch.stack(tuple(proposals.values()), dim=1) # [T, 3, N, H, W] 
    
    masked_nocs = [n[:, None] * m[None].to(n) for n, m in zip(gt_nocs, gt_masks)]


    # Reshape to fit box by performing roi_align
    W = proposals.shape[-1]  # Width of proposal we want gt to match
    targets = [project_on_boxes(m, p, i, W) 
        for m, p, i in zip(masked_nocs, proposed_box_regions, matched_ids)]
    targets = torch.cat(targets, dim=0) # [B, 3, H, W]

    # If target is empty return 0
    if targets.numel() == 0: return proposals.sum() * 0

    if loss_fx == cross_entropy:
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
    
    # TODO: make base class for all to inherit.
    elif isinstance(loss_fx, DiscriminatorWithOptimizer):
        loss = discriminator_as_loss(loss_fx, 
                                     proposals, 
                                     targets,
                                     reduction=reduction,
                                     mode=mode)
    elif isinstance(loss_fx, MultiClassDiscriminatorWithOptimizer):
        loss = multiclass_discriminator_as_loss(loss_fx, 
                                                proposals, 
                                                targets, 
                                                gt_labels,
                                                reduction=reduction,
                                                mode=mode)

    if dispersion_loss is not None:
        loss += dispersion_loss(proposals, targets) * dispersion_weight

    return loss