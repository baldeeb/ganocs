
def discretize_nocs(mast_gt_nocs):
# TODO ################################################
    pass

def nocs_class_loss(gt_mask, gt_nocs, pred_nocs):
# TODO ################################################
# NOTE: maybe instead of feeding in gt_nocs, we can 
#       feed in the discretized version of it.
    '''
    Takes in an instance segmentation mask along with
    the ground truth and predicted nocs maps. 

    Args: 
        gt_mask [B, C, H, W] (uint8): instance segmentation
        gt_nocs [B, 3, H, W] (uint8): ground truth nocs with 
                channel values 0-255.
        pred_nocs [B, 3, C, H, W] (bool): Discretized classification
                of the nocs map. C here indicates number of bins.
    
    returns [N]: A dictionary of lists containing loss values
    '''
    print('NOT YET IMPLEMENTED')
    pass