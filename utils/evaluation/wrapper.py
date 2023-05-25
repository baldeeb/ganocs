import torch
import numpy as np  
import wandb
from utils.evaluation.nocs_image_loss import l2_nocs_image_loss
from utils.visualization import draw_3d_boxes



def eval(model, dataloader, device, intrinsic, n_batches):
    with torch.no_grad():
        model.eval()
        loss = []
        for batch_i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            results = model(images)
            for result, target in zip(results, targets):
                loss.append(l2_nocs_image_loss(result['nocs'], 
                                          target['nocs'], 
                                          target['masks'],
                                          device=device))
            if n_batches is not None and batch_i >= n_batches:
                wandb.log({'eval_nocs_loss': sum(loss)/len(loss)})
                for result, image, target in zip(results, images, targets):
                    if np.prod(result['nocs'].shape) == 0: continue
                    img = draw_box(image*255.0, 
                                   result['masks'][0] > 0.5, 
                                   result['nocs'][0],
                                   np.array(target['depth']), 
                                   intrinsic)
                    _printable = lambda a: a.permute(1,2,0).detach().cpu().numpy()
                    wandb.log({'image': wandb.Image(img),
                            'nocs':  wandb.Image(_printable(result['nocs'][0])),})
                break
        model.train()
    

from utils.align import align
def draw_box(image, mask, nocs, depth, intrinsic):
    _to_ndarray = lambda a : a.clone().detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    transforms, scales, _ = align(_to_ndarray(mask), _to_ndarray(nocs), 
                                  _to_ndarray(depth), _to_ndarray(intrinsic))
    img = draw_3d_boxes(_to_ndarray(image.permute(1,2,0)), _to_ndarray(transforms[0]), 
                        _to_ndarray(scales[0]), _to_ndarray(intrinsic))
    return img
