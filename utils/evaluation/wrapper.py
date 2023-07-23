import torch
import numpy as np  
import wandb
from utils.evaluation.nocs_image_loss import l2_nocs_image_loss
from utils.visualization import draw_3d_boxes
from utils.align import align

def merge_nocs_to_single_image(nocs, masks, scores, 
                               score_threshold=0.5, 
                               overlap_threshold=0.1):
    '''
    Args:
        nocs: (B, 3, H, W)
        masks: (B, 1, H, W)
        scores: (B,)
        score_threshold: (float) minimum scores cosidered.
        overlap_threshold: (float) maximum overlap between boxes. 
            Boxes are process in order of score.
    '''
    nocs = nocs.permute(0,2,3,1)
    masks = masks.permute(0,2,3,1)

    # sort by score
    scores, indices = torch.sort(scores, descending=True)
    nocs = nocs[indices]; masks = masks[indices]

    merged = torch.zeros_like(nocs[0])
    merged_mask = torch.zeros_like(masks[0])
    for n, m, s in zip(nocs, masks, scores):
        if score_threshold and s < score_threshold: break
        if overlap_threshold and (merged_mask * m).sum() > (m.sum() * overlap_threshold):  continue
        merged_mask += m
        merged += n * m
    
    return (merged.clone().detach() * 255.0).int().cpu().numpy()


def draw_boxes(image, scores, masks, nocs, depth, intrinsic,
               score_threshold=0.9):
    img = (image*255.0).int().permute(1,2,0).detach().cpu().numpy()
    for score, mask, nocs in zip(scores, masks, nocs):
        if score < score_threshold: continue
        img = draw_box(img, mask > 0.5, nocs, np.array(depth), intrinsic)
    assert img is not None, 'Something went wront!'
    return img

def eval(model, dataloader, device, num_batches=None, log:callable=wandb.log):
    with torch.no_grad():
        score_threshold = 0.8

        model_training = model.training
        model.eval()
        for batch_i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            results = model(images)
            
            for result, image, target in zip(results, images, targets):
                if result['scores'].shape[0] == 0: # not objects detected
                    continue

                # Calculate some eval metrics
                loss = l2_nocs_image_loss(result['nocs'], 
                                          target['nocs'], 
                                          result['masks'],
                                          device=device)
                
                # Visualize Boxes
                img = draw_boxes(image, 
                                 result['scores'], 
                                 result['masks'], 
                                 result['nocs'], 
                                 target['depth'], 
                                 target['intrinsics'], 
                                 score_threshold)
                
                # Visualize NOCS
                cumm_nocs = merge_nocs_to_single_image(result['nocs'], 
                                                       result['masks'] > 0.5, 
                                                       result['scores'], 
                                                       score_threshold=score_threshold)
                
                log({
                    'eval_nocs_loss':   loss,
                    'image':            wandb.Image(img),
                    'pred_nocs':        wandb.Image(cumm_nocs),
                    'gt_nocs':          target['nocs'],
                })
            if num_batches is not None and batch_i >= num_batches: break
        if model_training: model.train()
    

def draw_box(image, mask, nocs, depth, intrinsic):
    _to_ndarray = lambda a : a.clone().detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    transforms, scales, _ = align(_to_ndarray(mask), _to_ndarray(nocs), 
                                  _to_ndarray(depth), _to_ndarray(intrinsic))
    img = draw_3d_boxes(image, _to_ndarray(transforms[0]), 
                        _to_ndarray(scales[0]), _to_ndarray(intrinsic))
    return img
