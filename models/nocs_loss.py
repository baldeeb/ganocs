import torch 
from torch.nn.functional import one_hot
from torchvision.ops import boxes as box_ops, roi_align
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def project_on_boxes(gt, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """
    NOTE: Code borrowed from 
    torchvision.models.detection.roi_heads.project_masks_on_boxes
    TODO: update description

    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt = gt.to(rois)
    return roi_align(gt, rois, (M, M), 1.0)

def nocs_loss(gt_labels, gt_nocs, nocs_proposals, mask_proposals, matched_ids):
    '''
    Takes in an instance segmentation mask along with
    the ground truth and predicted nocs maps. 

    Args: 
        nocs_proposals Dict[str, Tensor]: A dictionary of
                tensors containing the predicted nocs maps:
                - x [B, C, H, W] (float): x coordinate
                - y [B, C, H, W] (float): y coordinate
                - z [B, C, H, W] (float): z coordinate
        gt_nocs [B, 3, H, W] (float): ground truth nocs with
                values in [0, 1]
        

    
    returns [N]: A dictionary of lists containing loss values
    
    TODO: implement symmetry loss
    TODO: Experiment with
        - uing interpolation + relu as in relu grids and using an l2 loss
        -  
    '''
    proposals = torch.stack((nocs_proposals['x'], 
                             nocs_proposals['y'], 
                             nocs_proposals['z']), 
                             dim=1)

    labels = [gt_label[idxs] for gt_label, idxs 
              in zip(gt_labels, matched_ids)]
    labels = torch.cat(labels, dim=0)
    
    nocs_targets = [
        project_on_boxes(m, p, i, proposals.shape[-1]) 
        for m, p, i in zip(gt_nocs, mask_proposals, matched_ids)
    ]
    nocs_targets = torch.cat(nocs_targets, dim=0)
    nocs_targets = (nocs_targets * proposals.shape[2]).long()  # To indices
    targets = one_hot(nocs_targets, num_classes=proposals.shape[2])
    targets = targets.permute(0, 1, -1, 2, 3).float()
    
    if targets.numel() == 0: return proposals.sum() * 0

    return bce_loss(proposals, targets) 
