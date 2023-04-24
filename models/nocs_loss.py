import torch 
from torch.nn.functional import one_hot
from torchvision.ops import boxes as box_ops, roi_align
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.nn.functional import cross_entropy

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
    """
    NOTE: Code borrowed from 
    torchvision.models.detection.roi_heads.project_masks_on_boxes
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

def nocs_loss(gt_labels, gt_nocs, nocs_proposals, mask_proposals, matched_ids):
    '''
    Takes in an instance segmentation mask along with
    the ground truth and predicted nocs maps. 

    Args: 
        nocs_proposals Dict[str, Tensor]: A dictionary of
                tensors containing the predicted nocs maps:
                - x [B, C, N, H, W] (float): x coordinate
                - y [B, C, N, H, W] (float): y coordinate
                - z [B, C, N, H, W] (float): z coordinate
            Where C is the number of classes and N is bins.
        gt_nocs [3, H, W] (float): ground truth nocs with
                values in [0, 1]
        

    
    returns [N]: A dictionary of lists containing loss values
    
    TODO: implement symmetry loss
    TODO: Experiment with
        - uing interpolation + relu as in relu grids and using an l2 loss
    '''
    labels = [gt_label[idxs] for gt_label, idxs 
              in zip(gt_labels, matched_ids)]
    labels = torch.cat(labels, dim=0)
    l_idxs = torch.arange(labels.size(0))
    proposals = torch.stack((nocs_proposals['x'][l_idxs, labels], 
                             nocs_proposals['y'][l_idxs, labels], 
                             nocs_proposals['z'][l_idxs, labels]), 
                             dim=1)
    nocs_targets = [
        project_on_boxes(m, p, i, proposals.shape[-1]) 
        for m, p, i in zip(gt_nocs, mask_proposals, matched_ids)
    ]
    nocs_targets = torch.cat(nocs_targets, dim=0)
    nocs_targets = (nocs_targets * proposals.shape[2]).round().long()  # To indices
        
    if nocs_targets.numel() == 0: return proposals.sum() * 0
    
    return cross_entropy(proposals.transpose(1,2), nocs_targets)

