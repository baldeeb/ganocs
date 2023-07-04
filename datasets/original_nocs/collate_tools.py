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

    rgb = [torch.as_tensor(v[0]) for v in batch]

    targets = []
    for i, data in enumerate(batch):

        depth = torch.as_tensor(data[1])
        # image_metas     = inputs[1] ???
        # gt_class_ids    = data[2] ???
        intrinsics      = torch.as_tensor(data[2])
        
        labels          = torch.as_tensor(data[3])
        boxes           = torch.as_tensor(data[4])
        masks           = torch.as_tensor(data[5])
        nocs            = torch.as_tensor(data[6])

        ############################################################
        # NOTE: Guildeline to check the data types and shapes above.
        # masks = torch.as_tensor(labels2masks(images['semantics']))
        # boxes = mask2bbox(masks[None]).type(torch.float64).reshape([-1, 4])
        # labels = torch.ones(boxes.size(0)).type(torch.int64)  # NOTE: currently only one class exists
        # semantics = torch.as_tensor(images['semantics'].astype(np.int64)).unsqueeze(0)
        # semantic_ids = torch.as_tensor([v['semantic_id'] for v in meta['objects'].values()])
        ############################################################
        
        targets.append({
            'depth': depth, 
            'masks': masks, 
            'nocs': nocs,
            'labels': labels, 
            'boxes': boxes, 
            'camera_pose': None,
            'intrinsics': intrinsics,  # TODO: figure this out. 
            })

    return rgb, targets