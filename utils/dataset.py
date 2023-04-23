import torch 
import numpy as np

import numpy as np




import cv2 
import matplotlib.pyplot as plt 

# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, boxes):
    numpify = lambda a: a.clone().permute(1,2,0).detach().cpu().numpy()
    img = numpify(img)
    for a in boxes:
        x_min, y_min = int(a[0]), int(a[1])
        x_max, y_max = int(a[2]), int(a[3])
        color = (0, 255, 0)
        cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)
    plt.imsave('temp.png', img/img.flatten().max())








# def mask2bbox(mask, flatten=False):
#     '''
#     Takes in a mask of shape [???, H, W] and returns a tensor of shape [???, 4] 
#     where each element is a bounding box of the form [x, y, w, h] in the format
#     required by torchvision.ops.box_iou
#     '''
#     s = mask.shape[:-2] if len(mask.shape) > 2 else (1,)
#     H, W = mask.shape[-2:]
#     m = mask.reshape(-1, H, W)
#     idxs, i_s, j_s = np.where(m != 0)
#     bboxes, labels = [], []
#     for idx in np.unique(idxs):
#         i, j = i_s[idxs == idx], j_s[idxs == idx]
#         box = np.array([j.min(), i.min(), j.max(), i.max()])
#         labels.append(idx+1)
#         bboxes.append(box)
#     bboxes = np.stack(bboxes)
#     if flatten is False: bboxes = bboxes.reshape(*s, 4)
#     return bboxes, np.array(labels)




from torch import meshgrid, arange, stack, concatenate, Tensor
def mask2bbox(mask)->Tensor:
    '''
    Takes in a mask of shape [B, C, H, W] and returns a tensor of shape [B, C, 4] 
    where each element is a bounding box of the form [x, y, w, h] in the format
    required by torchvision.ops.box_iou
    '''
    _, _, H, W = mask.shape
    M = max(H, W)
    mgrid = stack(meshgrid(arange(H), arange(W)), dim=-1)  # [H, W, 2]
    mgrid = mgrid[None, None]                              # [1, 1, H, W, 2]
    mask = mask[:, :, :, :, None]                          # [B, C, H, W, 1]

    imask = mgrid*mask                                     # [B, C, H, W, 2]
    imask = imask.flatten(2, 3)                            # [B, C, HxW,  2]
    max_ij = imask.max(2).values                           # [B, C, 2]
    imask[imask == 0] = M
    min_ij = imask.min(2).values                           # [B, C, 2]
    
    boxes = concatenate([min_ij[:, :, 1], min_ij[:, :, 0], 
                         max_ij[:, :, 1], max_ij[:, :, 0],], dim=-1)          # [B, C, 4]
    return boxes




def labels2masks(labels, background=0):
    '''Creates a mask for each label in labels'''
    masks = []
    for label in np.unique(labels):
        if label == background: continue
        masks.append(labels == label)
    return np.stack(masks, axis=0)

# Load data    
def collate_fn(batch):
    def img2tensor(key):
        im = np.array([v[0][key]/255.0 for v in batch])
        return torch.as_tensor(im).permute(0, 3, 1, 2).float()
    # rgb = np.array([v[0]['image']/255.0 for v in batch])
    # rgb = torch.as_tensor(rgb).permute(0, 3, 1, 2).float()
    rgb = img2tensor('image')
    nocs = img2tensor('nocs')
    targets = []
    for i, data in enumerate(batch):
        images, meta = data[0], data[1]
        depth = torch.as_tensor(images['depth'])
        masks = torch.as_tensor(labels2masks(images['semantics']))
        boxes = mask2bbox(masks[None]).type(torch.float64).reshape([-1, 4])
        labels = torch.ones(boxes.size(0)).type(torch.int64)  # NOTE: currently only one class exists
        # semantics = torch.as_tensor(images['semantics'].astype(np.int64)).unsqueeze(0)
        # semantic_ids = torch.as_tensor([v['semantic_id'] for v in meta['objects'].values()])
        targets.append({
            'depth': depth, 'masks': masks, 'nocs': nocs[i],
            'labels': labels, 'boxes': boxes, 
            # 'semantic_ids': semantic_ids,
            })
        # draw_bounding_box(rgb[0]*255, boxes)

    return rgb, targets
