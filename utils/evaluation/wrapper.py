import torch
import numpy as np  
import wandb
from utils.evaluation.nocs_image_loss import l2_nocs_image_loss
from utils.evaluation.tools import iou
from utils.visualization import draw_3d_boxes
from utils.align import align
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes



def merge_masks(masks, overlap_threshold=0.5):
    merged_mask = torch.zeros_like(masks[0])
    for m in masks:
        if overlap_threshold and (merged_mask * m).sum() > (m.sum() * overlap_threshold): continue
        merged_mask += m

def merge_nocs_to_single_image(nocs, masks=None, overlap_threshold=0.1):
    '''
    Args:
        nocs: (B, 3, H, W)
        masks: (B, 1, H, W)
        scores: (B,)
        # score_threshold: (float) minimum scores cosidered.
        overlap_threshold: (float) maximum overlap between boxes. 
            Boxes are process in order of score.
    '''
    nocs = nocs.permute(0,2,3,1)
    if masks is not None: 
        masks = masks.permute(0,2,3,1)
        merged = torch.zeros_like(nocs[0])
        merged_mask = torch.zeros_like(masks[0])
        for n, m in zip(nocs, masks):
            if overlap_threshold and \
               (merged_mask * m).sum() > (m.sum() * overlap_threshold): 
                continue
            merged_mask += m
            merged += n * m
    else: 
        merged = torch.cat([n for n in nocs]).sum(dim=0) 
    
    return (merged.clone().detach() * 255.0).int().cpu().numpy()


def draw_boxes(image, masks, nocs, depth, intrinsic):
    img = (image*255.0).int().permute(1,2,0).detach().cpu().numpy()
    for mask, nocs in zip( masks, nocs):
        img = draw_box(img, mask, nocs, np.array(depth), intrinsic)
    assert img is not None, 'Something went wront!'
    return img

def eval(model, dataloader, device, num_batches=None, log:callable=wandb.log):
    with torch.no_grad():
        det_keys = ['nocs', 'masks', 'labels', 'boxes', 'scores']
        score_threshold = 0.8
        model_training = model.training
        model.eval()
        for batch_i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            results = model(images)
            
            for result, image, target in zip(results, images, targets):

                # Discard samples with low score
                # TODO: determine if this ought to be done in the model
                #       when using eval mode.
                select = result['scores'] > score_threshold
                if not any(select): continue
                for k in det_keys: result[k] = result[k][select]

                binary_masks = result['masks'] > 0.5

                # l2 nocs loss
                loss = l2_nocs_image_loss(result['nocs'], target['nocs'], 
                                          binary_masks, device=device)
                
                # 3d boxes
                img = draw_boxes(image, binary_masks, result['nocs'], 
                                 target['depth'], target['intrinsics'],)
                
                # nocs images
                cumm_nocs = merge_nocs_to_single_image(result['nocs'], binary_masks, )
                gt_nocs = target['nocs'].clone().detach().numpy().astype(int).transpose(1,2,0)

                # segmentation and 2d boxes
                int_image = (image.clone().detach() * 255.0).type(torch.uint8).cpu()
                seged_img = draw_segmentation_masks(int_image, binary_masks.squeeze(1).clone().detach().cpu(), alpha=0.6)
                seged_img = seged_img.permute(1,2,0).clone().detach().numpy().astype(int)
                boxes_img = draw_bounding_boxes(int_image, boxes=result['boxes'], width=4)
                boxes_img = boxes_img.permute(1,2,0).clone().detach().numpy().astype(int)
                
                # IoU
                IoUs = []
                for g, p in zip(target['boxes'][select], result['boxes'][select]):            
                    IoUs.append(iou(g, p, ))

                log({
                    'eval_nocs_loss':    loss,
                    'image':             wandb.Image(img),
                    'pred_nocs':         wandb.Image(cumm_nocs),
                    'gt_nocs':           wandb.Image(gt_nocs),
                    'pred_segmentation': wandb.Image(seged_img),
                    'pred_boxes':        wandb.Image(boxes_img),
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
