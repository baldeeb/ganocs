import torch
import numpy as np

# Load data    
def collate_fn(batch):
    # def img2tensor(key):
    #     im = np.array([v[0][key]/255.0 for v in batch])
    #     return torch.as_tensor(im).permute(0, 3, 1, 2).float()
    # # rgb = np.array([v[0]['image']/255.0 for v in batch])
    # # rgb = torch.as_tensor(rgb).permute(0, 3, 1, 2).float()
    # rgb = img2tensor('image')
    # nocs = img2tensor('nocs')

    # rgb = [torch.as_tensor(v[0].copy()).permute(2, 0, 1) for v in batch]
    rgb = []
    targets = []
    for i, data in enumerate(batch):

        depth           = torch.as_tensor(data[1]).float()
        intrinsics      = torch.as_tensor(data[2]).float()
        labels          = torch.as_tensor(data[3])
        b               = torch.as_tensor(data[4])[:, :4].float()  # 5 dim, last dim might be class or anchor
        boxes           = torch.stack([b[:, 1], b[:, 0], b[:, 3], b[:, 2]], dim=1) # width min, height min, width max, height max
        # boxes           = b
        masks           = torch.as_tensor(data[5]).permute(2, 0, 1).bool()
        nocs            = torch.as_tensor(data[6]).float().sum(dim=2).permute(2, 0, 1)
        
        if len(labels.shape) == 0 or labels.shape[0] == 0: 
            print(f'Warning: empty labels for image {i}')
            continue

        rgb.append(torch.as_tensor(data[0].copy()).permute(2, 0, 1))

        targets.append({
            'depth': depth, 
            'masks': masks, 
            'nocs': nocs,
            'labels': labels, 
            'boxes': boxes, 
            'camera_pose': None,
            'intrinsics': intrinsics,
        })

    rgb = torch.stack(rgb, dim=0).float()
    return rgb, targets