import torch 
import numpy as np
from utils.mask_to_bbox import mask2bbox


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
    for data in batch:
        images, meta = data[0], data[1]
        depth = torch.as_tensor(images['depth'])
        boxes, labels = mask2bbox(images['semantics'])
        boxes = torch.as_tensor(boxes.astype(np.int64))
        labels = torch.as_tensor(labels.astype(np.int64))
        masks = torch.as_tensor(labels2masks(images['semantics']))
        # semantics = torch.as_tensor(images['semantics'].astype(np.int64)).unsqueeze(0)
        # semantic_ids = torch.as_tensor([v['semantic_id'] for v in meta['objects'].values()])
        targets.append({
            'depth': depth, 'masks': masks, 'nocs': nocs,
            'labels': labels, 'boxes': boxes, 
            # 'semantic_ids': semantic_ids,
            })

    return rgb, targets
