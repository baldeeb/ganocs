import torch 

def l2_nocs_image_loss(pred_nocs, gt_nocs, pred_mask, device='cpu'):
    '''
    Args:
        pred_nocs: (N, 3, H, W)
        pred_mask: (N, H, W)
        gt_nocs:   (3, H, W)
    '''
    pred_nocs = pred_nocs.to(device)
    gt_nocs   = gt_nocs.to(device)
    pred_mask = pred_mask.to(device)
    pred      = pred_nocs * pred_mask
    gt        = gt_nocs[None] * pred_mask
    return torch.mean(torch.sqrt((pred - gt)**2, dim=1)).item()