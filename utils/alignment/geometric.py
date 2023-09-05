import torch
from torch import Tensor, logical_and, where, stack, concatenate


def valid_mask(depth:Tensor, masks:Tensor):
    return logical_and(masks, (depth > 0))

def depth_to_xyz(depth:Tensor, intrinsics:Tensor, masks:Tensor):
    idxs = where(masks)
    grid = stack([idxs[2], idxs[1]])            # [B, 2, W x H]
    ones = torch.ones([1, grid.shape[1]])
    uv_grid = concatenate((grid, ones), axis=0) # [B, 3, num_pixel]
    xyz = intrinsics.inverse() @ uv_grid        # [B, 3, num_pixel]
    xyz = xyz.T                                 # [B, num_pixel, 3]
    z = depth[idxs[1], idxs[2]]                 # [B, num_pixels]

    pts = xyz * z[:, None] / xyz[:, -1:]

    # Rotate 180 around z-axis # TODO: scrutinize
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]
    
    pts_batch = []
    for i in range(max(idxs[0])+1):
        s = torch.where(idxs[0] == i)
        pts_batch.append(pts[s])
    return pts_batch


def nocs_to_xyz(coord:Tensor, mask:Tensor):
    '''Expects a batch'''
    idxs = where(mask)
    coord_pts = coord[idxs[0], :, idxs[1], idxs[2]] - 0.5
    coord_batch = [coord_pts[where(idxs[0]==i)] for i in range(len(mask))]
    return coord_batch