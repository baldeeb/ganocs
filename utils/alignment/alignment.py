'''BYOC code TODO: officially reference'''

from typing import Optional

import torch

from .transformations import transform_points_Rt
from .geometric import valid_mask, depth_to_xyz, nocs_to_xyz
from .default_config import DEFAULT_ALIGN_CONFIG
from .weighted_procrustes import randomized_weighted_procrustes

def nn_gather(points, indices):
    # expand indices to same dimensions as points
    indices = indices[:, :, None]
    indices = indices.expand(indices.shape[0], indices.shape[1], points.shape[2])
    return points.gather(1, indices)


def align(corres, P, Q, align_cfg=DEFAULT_ALIGN_CONFIG(), overwrite_weighting=None):
    """
    Input:
        corres:  Information for K matches (list)
            idx_1   LongTensor(B, K)        match ids in pointcloud P
            idx_2   LongTensor(B, K)        match ids in pointcloud Q
            dists   FloatTensor(B, K)       match feature cosine distance
        P           FloatTensor (B, N, 3)   Source Pointcloud XYZ
        Q           FloatTensor (B, N, 3)   Target Pointcloud XYZ
        align_cfg: Alignment configuration (check config.py)

    Return:
        FloatTensor (B, 3, 4)       Rt matrix
        FloatTensor (B, )           Weighted Correspondance Error
    """
    # get useful variables
    corr_P_idx, corr_Q_idx, weights, _ = corres

    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx)
    corr_Q = nn_gather(Q, corr_Q_idx)

    Rt, _ = randomized_weighted_procrustes(corr_P, corr_Q, weights, align_cfg)

    # Calculate correspondance loss
    corr_P_rot = transform_points_Rt(corr_P, Rt)
    dist_PQ = (corr_P_rot - corr_Q).norm(p=2, dim=2)

    if overwrite_weighting is None:
        loss_weighting = align_cfg.loss_weighting
    else:
        loss_weighting = overwrite_weighting

    if loss_weighting == "none":
        corr_loss = dist_PQ.mean(dim=1)
    elif loss_weighting == "lowe":
        weights_norm = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        corr_loss = (weights_norm * dist_PQ).sum(dim=1)
    elif loss_weighting == "detached_lowe":
        weights_norm = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        corr_loss = (weights_norm * dist_PQ.detach()).sum(dim=1)
    else:
        raise ValueError(f"Unknown loss weighting: {loss_weighting}")

    return Rt, corr_loss, dist_PQ


def stretch_list_and_batch(data_list):
    B = len(data_list)
    N = max([len(e) for e in data_list])
    S = data_list[0].shape if len(data_list) > 0 else []
    batch = torch.zeros(B, N, *S[1:])
    for i in range(B):
        n = len(data_list[i])
        batch[i, :n] = data_list[i]
        n_comp = torch.randint(0, n, (N-n,))
        batch[i, n:N] = data_list[i][n_comp]
    return batch


def align_3d_points(P, Q, detach_transform=True):
    Rt, c = randomized_weighted_procrustes(P, Q, weights=None, align_cfg=DEFAULT_ALIGN_CONFIG())
    if detach_transform: Rt = Rt.detach()
    
    # Calculate correspondance loss
    corr_P_rot = transform_points_Rt(P, Rt)
    dist_PQ = (corr_P_rot - Q).norm(p=2, dim=2)
    corr_loss = dist_PQ.mean(dim=1)
    
    # Rts = c * Rt
    # return Rts, corr_loss, dist_PQ

    return Rt, c, corr_loss

def align_nocs(masks:torch.Tensor, coords:torch.Tensor, 
               depth:torch.Tensor, intrinsics:torch.Tensor,):
    '''masks, coords [B x ...]'''
    m = valid_mask(depth, masks)
    source_list = nocs_to_xyz(coords, m)
    source = stretch_list_and_batch(source_list)
    target_list = depth_to_xyz(depth, intrinsics, m) 
    target = stretch_list_and_batch(target_list)
    print(f'source {source.shape}, target {target.shape}')
    return align_3d_points(source, target)
