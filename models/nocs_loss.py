import torch 
from torchvision.ops import boxes as _, roi_align
from torch.nn.functional import (cross_entropy, 
                                 binary_cross_entropy, 
                                 softmax)
from models.nocs_util import select_labels
from models.discriminator import DiscriminatorWithOptimizer
import cv2 
import matplotlib.pyplot as plt 

# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, boxes):
    numpify = lambda a: a.clone().permute(1,2,0).detach().cpu().numpy()
    img = numpify(img)
    for a in boxes:
        x_min, y_min = int(a[0]), int(a[1])
        x_max, y_max = int(a[2]), int(a[3])
        color = (0, 255, 0)
        cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)
    plt.imsave('temp.png', img/img.flatten().max())

def project_on_boxes(gt, boxes, matched_idxs, M):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    """ Code borrowed from torchvision.models.detection.roi_heads.project_masks_on_boxes
    
    TODO: update description

    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Args:
        gt (Tensor[C, H, W]): Each element contains ``C`` feature maps 
            of dimensions ``H x W``.
        boxes (Tensor[N, 4]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
        matched_idxs (Tensor[N]): 
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt = gt.unsqueeze(0).to(rois)
    return roi_align(gt, rois, (M, M), 1.0)

def discriminator_as_loss(discriminator:DiscriminatorWithOptimizer, 
                          proposals:torch.Tensor, 
                          targets:torch.Tensor,
                          reduction:str='mean'):
    '''
    Args:
        proposals (Tensor): [B, 3, N, H, W] tensor of predicted nocs
            where B is batch size, 3 is for x,y,z channels, N is the
            number of bins, and H, W are the height and width of the
            image.
        targets (Tensor): [B, 3, H, W] tensor of indices indicating
            which of the binary logits is the correct nocs value.'''
    
    # Get weighted average of proposal
    N = proposals.shape[2]
    bin_idxs = torch.arange(N).to(proposals)
    bin_idxs = bin_idxs[None, None, :, None, None]
    soft_nocs = softmax(proposals, dim=2)
    weighted_avg = (soft_nocs * bin_idxs).sum(dim=2) / N

    # Train discriminator
    discriminator.train_step(targets, weighted_avg)
    
    # Get nocs loss
    l = discriminator(weighted_avg)
    return binary_cross_entropy(l, torch.ones_like(l), reduction=reduction)

    

def nocs_loss(gt_labels, 
              gt_nocs, 
              nocs_proposals, 
              proposed_box_regions, 
              matched_ids, 
              reduction='mean', 
              loss_fx=cross_entropy):
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
    Returns 
        loss (Tensor): An element or list of values depending on the reduction
    
    NOTE: Symmetry loss is not implemented
    '''

    # Select the label for each proposal
    labels = [gt_label[idxs] for gt_label, idxs 
              in zip(gt_labels, matched_ids)]
    proposals = select_labels(nocs_proposals, labels)  # Dict (3 values) [B, N, H, W] 
    proposals = torch.stack(tuple(proposals.values()), dim=1) # [B, 3, N, H, W] 
    
    # Reshape to fit box by performing roi_align
    W = proposals.shape[-1]  # Width of proposal we want gt to match
    targets = [project_on_boxes(m, p, i, W) 
        for m, p, i in zip(gt_nocs, proposed_box_regions, matched_ids)]
    targets = torch.cat(targets, dim=0) # [B, 3, H, W]

    # If target is empty return 0
    if targets.numel() == 0: return proposals.sum() * 0

    if loss_fx == cross_entropy:
        targets = (targets * proposals.shape[2]).round().long()  # To indices
        return cross_entropy(proposals.transpose(1,2), 
                             targets,
                             reduction=reduction)
    elif isinstance(loss_fx, DiscriminatorWithOptimizer):
        return discriminator_as_loss(loss_fx, proposals, targets,
                                     reduction=reduction)

