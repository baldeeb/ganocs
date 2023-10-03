import torch 

def l2_nocs_image_loss(pred_nocs, gt_nocs, pred_mask, labels=None, device='cpu'):
    ################################################
    
    s = {
        'bottle': [0, 0.3, 0],
        'cup':    [0, 0.3, 0],
        'mug':    [0, 0.3, 0],
    }
    all_classes = [ 'BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug',]
    classes = [all_classes[i] for i in labels]
    sym_T = [rot_mats_from_angle_axis(torch.tensor(s.get(c, [0.0, 0.0, 0.0]))) for c in classes]
    ################################################
    '''
    Args:
        pred_nocs: (N, 3, H, W)
        pred_mask: (N, H, W)
        gt_nocs:   (3, H, W)
    '''
    pred_nocs = pred_nocs.to(device)
    gt_nocs   = gt_nocs.to(device)
    pred_mask = pred_mask.to(device).bool()
    pred      = pred_nocs * pred_mask
    gts        = gt_nocs[None] * pred_mask
    # gts = [rotate_nocs(n, T) for n, T in zip(gts, sym_T)]
    # b, _, i, j = tuple(torch.where(pred_mask))
    deltas = []
    for p, g, T, m in zip(pred, gts, sym_T, pred_mask):
        _, i, j = torch.where(m)
        gs = rotate_nocs(g, T)
        deltas.append((p[None]-gs)[:, :, i, j].square().sqrt().mean((-1,-2)).min().item())

    return torch.tensor(deltas).mean().item()
    # delta = (pred - gts)[b, :, i, j]
    # return delta.square().sqrt().mean().item()


from pytorch3d.transforms import axis_angle_to_matrix

def rot_mats_from_angle_axis(axis_angle_increment):
    deg = torch.linalg.norm(axis_angle_increment)
    if deg == 0: return torch.eye(3)[None]
    inc = torch.arange((int( 2 * torch.pi / deg )))
    axis = axis_angle_increment / deg
    axis_angles = inc[:, None] * axis[None]
    return axis_angle_to_matrix(axis_angles)


def rotate_nocs(nocs, rotations):
    '''
    Args:
        nocs (torch.Tensor): [3, H, W]
        rotations (torch.Tensors): [M, 3, 3] SO(3) transforms
    '''
    rotations = rotations.to(nocs.device)
    _, H, W = nocs.shape
    M, _, _ = rotations.shape
    _, i, j = torch.where(nocs != 0)
    pts = nocs[:, i, j] - 0.5
    t_pts = rotations @ pts                             
    t_nocs = torch.zeros(M, 3, H, W).to(nocs.device)   
    t_nocs[:, :, i, j] = t_pts + 0.5
    return t_nocs