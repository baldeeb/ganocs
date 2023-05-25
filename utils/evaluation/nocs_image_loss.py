import torch 

def l2_nocs_image_loss(pred_nocs, gt_nocs, gt_masks, device='cpu'):
    '''
    Args:
        pred_nocs: (N, 3, H, W)
        gt_nocs: (3, H, W)
        gt_mask: (N, H, W)'''
    pred_nocs =  pred_nocs.to(device)
    gt_nocs   =  gt_nocs.to(device)
    gt_masks  =  gt_masks.to(device)
    pred = pred_nocs * gt_masks[:, None]
    gt = gt_nocs[None] * gt_masks[:, None]
    return torch.mean(torch.norm(pred - gt, dim=1)).item()