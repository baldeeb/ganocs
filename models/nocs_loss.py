from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discretize_nocs(mast_gt_nocs):
# TODO ################################################
    pass

def nocs_loss(gt_mask, gt_nocs, pred_nocs):
    '''
    Takes in an instance segmentation mask along with
    the ground truth and predicted nocs maps. 

    Args: 
        gt_mask [B, C, H, W] (uint8): instance segmentation
        gt_nocs [B, 3, H, W] (uint8): ground truth nocs with 
                channel values 0-255.
        pred_nocs [B, 3, C, H, W] (float): Softmaxed binary
                prediction. C here indicates number of bins.
    
    returns [N]: A dictionary of lists containing loss values
    
    TODO: implement symmetry loss
    TODO: Experiment with
        - uing interpolation + relu as in relu grids and using an l2 loss
        -  
    '''

    masked_gt = gt_nocs.unsqueeze(2) * gt_mask.unsqueeze(1)
    loss = bce_loss(pred_nocs, masked_gt) 
    return loss
