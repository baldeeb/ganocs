import numpy as np
def mask2bbox(mask, flatten=False):
    '''
    Takes in a mask of shape [???, H, W] and returns a tensor of shape [???, 4] 
    where each element is a bounding box of the form [x, y, w, h] in the format
    required by torchvision.ops.box_iou
    '''
    s = mask.shape[:-2] if len(mask.shape) > 2 else (1,)
    H, W = mask.shape[-2:]
    m = mask.reshape(-1, H, W)
    idxs, i_s, j_s = np.where(m != 0)
    bboxes = []
    for idx in np.unique(idxs):
        i, j = i_s[idxs == idx], j_s[idxs == idx]
        i_ext = np.array([i.min(), i.max()])
        j_ext = np.array([j.min(), j.max()])
        bboxes.append(np.array([j_ext[0], i_ext[0], j_ext[1]-j_ext[0], i_ext[1]-i_ext[0]]))
    bboxes = np.stack(bboxes)
    if flatten is False: bboxes = bboxes.reshape(*s, 4)
    return bboxes