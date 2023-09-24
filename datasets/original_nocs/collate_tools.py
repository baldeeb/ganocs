import torch
import numpy as np

# TODO: return numpy arrays instead of torch tensors
def collate_fn(batch):
    rgb = []
    targets = []
    for _, data in enumerate(batch):

        depth           = data[1]
        # depth           = torch.as_tensor(data[1]).float()

        if data[2] is None: intrinsics = None
        else: intrinsics = torch.as_tensor(data[2]).float()
        
        labels          = torch.as_tensor(data[3]).type(torch.int64)
        b               = torch.as_tensor(data[4])[:, :4].float()  # 5 dim, last dim might be class or anchor
        boxes           = torch.stack([b[:, 1], b[:, 0], b[:, 3], b[:, 2]], dim=1) # width min, height min, width max, height max
        masks           = torch.as_tensor(data[5]).permute(2, 0, 1).bool()
        nocs            = torch.as_tensor(data[6]).float().sum(dim=2).permute(2, 0, 1)
        ignore_nocs     = torch.as_tensor(data[7]).float()
        scales          = torch.as_tensor(data[8]).float()

        if len(labels.shape) == 0 or labels.shape[0] == 0: 
            raise RuntimeError(f'Warning: collate called on data with no labels.')

        rgb.append(torch.as_tensor(data[0].copy()).permute(2, 0, 1))

        targets.append({
            'depth': depth, 
            'masks': masks, 
            'nocs': nocs,
            'labels': labels, 
            'boxes': boxes, 
            'scales': scales,
            'camera_pose': None,
            'intrinsics': intrinsics,
            'no_nocs': ignore_nocs,
        })

    # rgb = torch.stack(rgb, dim=0).float()
    return rgb, targets