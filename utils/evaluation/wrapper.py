import torch
import numpy as np  
import wandb
from utils.evaluation.nocs_image_loss import l2_nocs_image_loss
from utils.evaluation.tools import iou, compute_mAP,compute_ap_from_matches_scores
from utils.visualization import draw_3d_boxes
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from tqdm import tqdm

from utils.align import align 
# from utils.alignment import align  # TODO: use this instead


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
    
    if len(nocs) == 0:
        merged = torch.zeros((1, *nocs.shape[1:])).int().cpu().numpy()
    else:
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
        merged = (merged.clone().detach() * 255.0).int().cpu().numpy()
    return merged


def draw_box(image, Rt, s, intrinsic):
    _to_ndarray = lambda a : a.clone().detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    img = draw_3d_boxes(image, Rt[0], s[0], _to_ndarray(intrinsic))
    return img

def draw_boxes(image, Rts, Ss, intrinsic):
    img = (image*255.0).int().permute(1,2,0).detach().cpu().numpy()
    for Rt, s in zip(Rts, Ss):
        img = draw_box(img, Rt, s, intrinsic)
    assert img is not None, 'Something went wront!'
    return img

def eval(model, dataloader, device, log_mAP_vals,synset_names,num_batches=None, log:callable=wandb.log):
    with torch.no_grad():
        model_training = model.training
        model.eval()
        if num_batches is None: num_batches = len(dataloader)
        if log_mAP_vals.do_log:
            num_classes = len(synset_names)
            degree_thresholds = range(
                log_mAP_vals.get('degree_thresholds_start', 0), 
                log_mAP_vals.get('degree_thresholds_end', 61), 
                log_mAP_vals.get('degree_thresholds_step', 1)
            )
            shift_thresholds = np.linspace(
                log_mAP_vals.get('shift_thresholds_start', 0), 
                log_mAP_vals.get('shift_thresholds_end', 1), 
                log_mAP_vals.get('shift_thresholds_steps', 31)
            ) * log_mAP_vals.get('shift_thresholds_multiplier', 15)
            iou_3d_thresholds = np.linspace(
                log_mAP_vals.get('iou_3d_thresholds_start', 0), 
                log_mAP_vals.get('iou_3d_thresholds_end', 1), 
                log_mAP_vals.get('iou_3d_thresholds_steps', 101)
            )
            degree_thres_list = list(degree_thresholds) + [360]
            num_degree_thres = len(degree_thres_list)

            shift_thres_list = list(shift_thresholds) + [100]
            num_shift_thres = len(shift_thres_list)

            iou_thres_list = list(iou_3d_thresholds)
            num_iou_thres = len(iou_thres_list)
            iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
            iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
            iou_pred_scores_all  = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
            iou_gt_matches_all   = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
            
            pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
            pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
            pose_gt_matches_all  = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
            pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
        for batch_i, (images, targets) in tqdm(enumerate(dataloader), 
                                               total=num_batches,
                                               leave=False, desc='Eval Loop'):
            images = [img.to(device) for img in images]
            results = model(images)
            
            for result, image, target in zip(results, images, targets):

                pred_bin_masks = result['masks'] > 0.5
                
                # String descriptions
                descriptions = [f'cls: {l.item()} - {s.item()}%' \
                                for l, s in zip(result['labels'], 
                                                result['scores'])]

                # segmentation and 2d boxes
                int_image = (image.clone().detach() * 255.0).type(torch.uint8).cpu()
                seged_img = draw_segmentation_masks(int_image, pred_bin_masks.squeeze(1).clone().detach().cpu(), alpha=0.6)
                seged_img = seged_img.permute(1,2,0).clone().detach().numpy().astype(int)
                boxes_img = draw_bounding_boxes(int_image, boxes=result['boxes'], 
                                                width=4, labels=descriptions )
                boxes_img = boxes_img.permute(1,2,0).clone().detach().numpy().astype(int)
                
                log_results = { # MRCNN
                    'pred_segmentation': wandb.Image(seged_img),
                    'pred_boxes':        wandb.Image(boxes_img),
                }                    

                if callable(getattr(model.roi_heads, "has_nocs", None)) and \
                    model.roi_heads.has_nocs():
                    # l2 nocs loss
                    loss = l2_nocs_image_loss(result['nocs'], target['nocs'], 
                                            pred_bin_masks, device=device)
                    
                    # align
                    # TODO: Move to the GPU driven align function implemented by BYOC
                    _to_ndarray = lambda a : a.clone().detach().cpu().numpy() \
                                            if isinstance(a, torch.Tensor) else a
                    pred_Rt, pred_s = [], []
                    for mask, nocs in zip(_to_ndarray(pred_bin_masks), 
                                        _to_ndarray(result['nocs'])):
                        Rt, s, _ = align(mask, nocs, 
                                        _to_ndarray(target['depth']), 
                                        _to_ndarray(target['intrinsics']))
                        pred_Rt.append(Rt), pred_s.append(s)
                    result["pred_Rt"]=pred_Rt
                    result["scales"]=pred_s

                    gt_Rt, gt_s = [], []
                    for mask in _to_ndarray(target['masks']):
                        mask = mask[None]
                        nocs = _to_ndarray(target['nocs'] * mask)
                        Rt, s, _ = align(mask, nocs, 
                                         _to_ndarray(target['depth']), 
                                         _to_ndarray(target['intrinsics']))
                        gt_Rt.append(Rt), gt_s.append(s)
                    target["gt_Rt"]=gt_Rt
                  
                    # 3d boxes
                    # TODO this process, like IOU calculations need to get 3D boxes. Avoid redundant calculations.
                    pred_bboxes_image = draw_boxes(image, 
                                                   _to_ndarray(pred_Rt), 
                                                   _to_ndarray(pred_s), 
                                                   _to_ndarray(target['intrinsics']),)
                    gt_bboxes_image   = draw_boxes(image, 
                                                   _to_ndarray(gt_Rt),   
                                                   _to_ndarray(gt_s),   
                                                   _to_ndarray(target['intrinsics']),)
                    
                    # nocs images
                    cumm_nocs = merge_nocs_to_single_image(result['nocs'], pred_bin_masks, )
                    gt_nocs = (target['nocs']*255).clone().detach().numpy().astype(int).transpose(1,2,0)

                    log_results.update({
                        'nocs_l2_loss':      loss,
                        'pred_bboxes':       wandb.Image(pred_bboxes_image),
                        'gt_bboxes':         wandb.Image(gt_bboxes_image),
                        'pred_nocs':         wandb.Image(cumm_nocs),
                        'gt_nocs':           wandb.Image(gt_nocs),
                    })

                    # IoU
                    if len(result['boxes']) > 0 and len(target['boxes']) > 0:
                        matched_idxs, labels = model.roi_heads.assign_targets_to_proposals(
                                                                        [result['boxes'].cpu()], 
                                                                        [target['boxes'].cpu()], 
                                                                        [target['labels'].cpu()])
                        iou_vals = []
                        # TODO: batchify this
                        for gt_i, pred_i in zip(matched_idxs[0], range(len(matched_idxs[0]))):
                            if pred_Rt[pred_i].sum() == 0: continue
                            iou_vals.append(
                                iou(gt_Rt[gt_i][0], pred_Rt[pred_i][0], 
                                    gt_s[gt_i][0],  pred_s[pred_i][0]))
                        log_results['IoU'] = np.mean(iou_vals)

                log(log_results)
                if log_mAP_vals.do_log:
                    compute_mAP(result,target,device,synset_names,iou_thres_list,iou_pred_matches_all,iou_pred_scores_all,
                                iou_gt_matches_all,log_mAP_vals.use_matches_for_pose,log_mAP_vals.iou_pose_thres,degree_thres_list, shift_thres_list,
                                pose_pred_matches_all,pose_pred_scores_all,pose_gt_matches_all)
                    


            if num_batches is not None and batch_i >= num_batches: break
            if log_mAP_vals.do_log:
                iou_dict = {}
                iou_dict['thres_list'] = iou_thres_list
                for cls_id in range(1, num_classes):
                    class_name = synset_names[cls_id]
                    for s, iou_thres in enumerate(iou_thres_list):
                        iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
                                                                            iou_pred_scores_all[cls_id][s, :],
                                                                            iou_gt_matches_all[cls_id][s, :])    

                iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
                
                iou_dict['aps'] = iou_3d_aps

                for i, degree_thres in enumerate(degree_thres_list):                
                    for j, shift_thres in enumerate(shift_thres_list):
                        # print(i, j)
                        for cls_id in range(1, num_classes):
                            cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                            cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                            cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                            pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all, 
                                                                                    cls_pose_pred_scores_all, 
                                                                                    cls_pose_gt_matches_all)

                        pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
                

                iou_aps = iou_3d_aps
                print('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.25)] * 100))
                print('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.5)] * 100))
                print('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(5)] * 100))
                print('5 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(100)] * 100))
                print('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(5)] * 100))
                print('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(10)] * 100))
                print('15 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(5)] * 100))
                print('15 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(10)] * 100))
                log_results["3D IoU at 25"]=iou_aps[-1, iou_thres_list.index(0.25)] * 100
                log_results["3D IoU at 50"]=iou_aps[-1, iou_thres_list.index(0.5)] * 100
                log_results["5 degree, 5cm"]=pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(5)] * 100
                log_results["5 degree, 10cm"]=pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(10)]*100
                log_results["10 degree, 5cm"]=pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(5)] * 100
                log_results["10 degree, 10cm"]=pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(10)] * 100
                log_results["15 degree, 5cm"]=pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(5)] * 100
                log_results["15 degree, 10cm"]=pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(10)] * 100
                
                log(log_results)



        if model_training: model.train()
    



# def align_targets(masks:torch.Tensor, coords:torch.Tensor, depth:torch.Tensor, intrinsics:torch.Tensor):
#     ''' Alings a bounding box to all detections in an image.
#     The first index of each of coords, masks, and depth is expected to indicate the 
#     number of instances that is being processed.
#     Args: 
#         masks  [B, H, W] is a set of masks one per object in view. 
#         coords [3, H, W] is an image of NOCS coordinates.
#         depth  [   H, W] depth associated with the NOCS image.
#         intrinsic [3, 3] is the camera intrinsic matrix.
#         '''
#     if masks.shape[0] > 0: raise RuntimeError('Attempting to align zero detections...')
#     transforms    = np.zeros((num_instances, 4, 4))
#     scales        = np.ones((num_instances, 3))
#     for i, mask in enumerate(masks):
#         Rt, corr_loss, dist_PQ = align(mask, coords, depth, intrinsics)
#     return transforms, scales

