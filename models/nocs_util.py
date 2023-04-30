from torch import cat, arange, Tensor, stack
def select_labels(nocs_dict, 
                  labels):
    '''
    Args:
        data (torch.Tensor): of shape [B, L, ...] where B
            is the batch size L is the labels one of which
            we need to retain.
        labels (List[int]): containing the supported labels
    '''
    l = cat(labels) - 1
    ids = arange(l.size(0), device=l.device)
    discard_ids = arange(l.size(0), device=l.device)[l==-1]
    l[l==-1] = 0
    for _, (k, v) in enumerate(nocs_dict.items()):
        v = v[ids, l]
        v[discard_ids] = v[discard_ids] * 0
        nocs_dict[k] = v
    return nocs_dict

def rotate_nocs(nocs:Tensor, 
                masks:Tensor, 
                rotation:Tensor, 
                times:int=1
            )->Tensor:
    '''
    Args:
        nocs [B, 3, H, W]
        mask [B, 1, H, W] of boolean dtype.
        rotation [*, 3, 3] where * can be 1 or B.
        times indicates the number of times to rotate
    Returns: 
        A rotated version of the nocs mask. If times==1
        the shape is retained, otherwise the returned 
        tensor will be of shape [B, times, 3, H, W].
    '''
    rots = [rotation]
    for _ in range(times-1):
        rots.append(rotation @ rots[-1])
    rots = stack(rots, dim=1)

    B, C, H, W = nocs.shape
    r =  rots @ nocs[:, None].flatten(-2,-1)
    r = r.reshape(B, times, C, H, W) * masks[:, None]
    return r.squeeze(1)