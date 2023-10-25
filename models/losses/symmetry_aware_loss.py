import torch 
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix

class SymmetryAwareLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, *args, **kwargs):
        return symmetry_aware_loss(*args, **kwargs)

def symmetry_aware_loss(pred_nocs, gt_nocs, **kwargs):
    '''
    Args:
        pred_nocs: (N, 3, H, W)
        gt_nocs:   (N, 3, H, W)
    '''
    sym_T = get_object_symmetry_rotations(kwargs['labels'])
    deltas = []
    for p, g, T in zip(pred_nocs, gt_nocs, sym_T):
        m = torch.zeros_like(g); m[g > 1e-6] = 1.0
        gs = (torch.einsum('ijj,jkl->ijkl', T.to(g), (g - 0.5)) + 0.5)*m
        #gs=torch.unsqueeze(g,0)
        if kwargs['loss_type']=='mse':
            deltas.append((p[None] - gs.detach()).view(len(gs), -1).square().mean(-1).min())
        elif kwargs['loss_type']=='cross_entropy':
            #p=torch.unsqueeze(p,0)
            p_broadcasted = p.unsqueeze(0).repeat(gs.size(0), 1, 1, 1, 1)
            gs_discretized = (gs * (p_broadcasted.shape[2] - 1)).round().long()  # (0->1) to indices [0, 1, ...]
            loss = F.cross_entropy(p_broadcasted.transpose(1,2), gs_discretized, reduction=kwargs['reduction']).min()
            deltas.append(loss)
    return torch.stack(deltas).mean() if len(deltas) > 0 else torch.tensor(0.0)

def rot_mats_from_angle_axis(axis_angle_increment):
    deg = torch.linalg.norm(axis_angle_increment)
    if deg == 0: return torch.eye(3)[None]
    inc = torch.arange((int( 2 * torch.pi / deg )))
    axis_angles = inc[:, None] * axis_angle_increment[None]
    return axis_angle_to_matrix(axis_angles)

def get_object_symmetry_rotations(labels):
    s = {
        'bottle': [0, 0.3, 0],
        'cup':    [0, 0.3, 0],
        'bowl':    [0, 0.3, 0],
    }
    all_classes = [ 'BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug',]
    classes = [all_classes[i] for i in labels]
    sym_T = [rot_mats_from_angle_axis(torch.tensor(s.get(c, [0.0, 0.0, 0.0]))) for c in classes]
    return sym_T
