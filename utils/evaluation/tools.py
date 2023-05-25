"""
Cropped and edited from util.py of the original NOCS repo.

previously noted:
    Mask R-CNN
    Common utility functions and classes.
    Copyright (c) 2017 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Waleed Abdulla
"""

import numpy as np
from pytorch3d.ops import box3d_overlap

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def asymmetric_3d_iou(RT1, RT2, scales1, scales2):
    '''
    Args:
        RT{1, 2}:     [{M, N}, 4, 4]
        scales{1, 2}: [{M, N}, 3]
    '''
    cube1 = get_3d_bbox(scales1, 0)
    bbox1 = transform_coordinates_3d(cube1, RT1)
    cube2 = get_3d_bbox(scales2, 0)
    bbox2 = transform_coordinates_3d(cube2, RT2)
    _, iou = box3d_overlap(bbox1, bbox2)
    return iou

def iou(RT_1, RT_2, scales_1, scales_2):
    """
    Input: 
        RT_1: [4, 4]
        RT_2: [4, 4]
        scales_1: [3]
        scales_2: [3]
    Return 
        iou: scalar
    """
    iou_3d = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)
    return iou_3d



# def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
#     noc_cube_1 = get_3d_bbox(scales_1, 0)
#     bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

#     noc_cube_2 = get_3d_bbox(scales_2, 0)
#     bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

#     bbox_1_max = np.amax(bbox_3d_1, axis=0)
#     bbox_1_min = np.amin(bbox_3d_1, axis=0)
#     bbox_2_max = np.amax(bbox_3d_2, axis=0)
#     bbox_2_min = np.amin(bbox_3d_2, axis=0)

#     overlap_min = np.maximum(bbox_1_min, bbox_2_min)
#     overlap_max = np.minimum(bbox_1_max, bbox_2_max)

#     # intersections and union
#     if np.amin(overlap_max - overlap_min) <0:
#         intersections = 0
#     else:
#         intersections = np.prod(overlap_max - overlap_min)
#     union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
#     overlaps = intersections / union
#     return overlaps


# def compute_3d_matches(gt_class_ids, gt_RTs, gt_scales, 
#                        pred_boxes, pred_class_ids, 
#                        pred_scores, pred_RTs, pred_scales,
#                        iou_3d_thresholds, score_threshold=0):
#     """Finds matches between prediction and ground truth instances.
#     Returns:
#         gt_matches: 2-D array. For each GT box it has the index of the matched
#                   predicted box.
#         pred_matches: 2-D array. For each predicted box, it has the index of
#                     the matched ground truth box.
#         overlaps: [pred_boxes, gt_boxes] IoU overlaps.
#     """
#     num_pred = len(pred_class_ids)
#     num_gt = len(gt_class_ids)
#     indices = np.zeros(0)
    
#     if num_pred:
#         # pred_boxes = trim_zeros(pred_boxes).copy()
#         pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

#         # Sort predictions by score from high to low
#         indices = np.argsort(pred_scores)[::-1]
        
#         pred_boxes = pred_boxes[indices].copy()
#         pred_class_ids = pred_class_ids[indices].copy()
#         pred_scores = pred_scores[indices].copy()
#         pred_scales = pred_scales[indices].copy()
#         pred_RTs = pred_RTs[indices].copy()

#     # Compute IoU overlaps [pred_bboxs gt_bboxs]
#     #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
#     overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
#     for i in range(num_pred):
#         for j in range(num_gt):
#             overlaps[i, j] = iou(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j])

#     # Loop through predictions and find matching ground truth boxes
#     num_iou_3d_thres = len(iou_3d_thresholds)
#     pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
#     gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

#     for s, iou_thres in enumerate(iou_3d_thresholds):
#         for i in range(len(pred_boxes)):
#             # Find best matching ground truth box
#             # 1. Sort matches by score
#             sorted_ixs = np.argsort(overlaps[i])[::-1]
#             # 2. Remove low scores
#             low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
#             if low_score_idx.size > 0:
#                 sorted_ixs = sorted_ixs[:low_score_idx[0]]
#             # 3. Find the match
#             for j in sorted_ixs:
#                 # If ground truth box is already matched, go to next one
#                 #print('gt_match: ', gt_match[j])
#                 if gt_matches[s, j] > -1:
#                     continue
#                 # If we reach IoU smaller than the threshold, end the loop
#                 iou = overlaps[i, j]
#                 #print('iou: ', iou)
#                 if iou < iou_thres:
#                     break
#                 # Do we have a match?
#                 if not pred_class_ids[i] == gt_class_ids[j]:
#                     continue

#                 if iou > iou_thres:
#                     gt_matches[s, j] = i
#                     pred_matches[s, i] = j
#                     break

#     return gt_matches, pred_matches, overlaps, indices


# def compute_degree_cm_mAP(results, degree_thresholds=[360], shift_thresholds=[100], iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False):
#     """Compute Average Precision at a set IoU threshold (default 0.5).
#     Returns:
#     mAP: Mean Average Precision
#     precisions: List of precisions at different class score thresholds.
#     recalls: List of recall values at different class score thresholds.
#     overlaps: [pred_boxes, gt_boxes] IoU overlaps.
#     """
    
#     num_classes = len(synset_names)
#     degree_thres_list = list(degree_thresholds) + [360]
#     num_degree_thres = len(degree_thres_list)

#     shift_thres_list = list(shift_thresholds) + [100]
#     num_shift_thres = len(shift_thres_list)

#     iou_thres_list = list(iou_3d_thresholds)
#     num_iou_thres = len(iou_thres_list)

#     if use_matches_for_pose:
#         assert iou_pose_thres in iou_thres_list

#     iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
#     iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
#     iou_pred_scores_all  = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
#     iou_gt_matches_all   = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    
#     pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
#     pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
#     pose_gt_matches_all  = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
#     pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]

#     # loop over results to gather pred matches and gt matches for iou and pose metrics
#     progress = 0
#     for progress, result in enumerate(results):
        
#         gt_class_ids = result['gt_class_ids'].astype(np.int32)
#         gt_RTs = np.array(result['gt_RTs'])
#         gt_scales = np.array(result['gt_scales'])
#         gt_handle_visibility = result['gt_handle_visibility']
    
#         pred_bboxes = np.array(result['pred_bboxes'])
#         pred_class_ids = result['pred_class_ids']
#         pred_scales = result['pred_scales']
#         pred_scores = result['pred_scores']
#         pred_RTs = np.array(result['pred_RTs'])
#         #print(pred_bboxes.shape[0], pred_class_ids.shape[0], pred_scores.shape[0], pred_RTs.shape[0])

#         if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
#             continue


#         for cls_id in range(1, num_classes):
#             # get gt and predictions in this class
#             cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
#             cls_gt_scales = gt_scales[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            
#             cls_gt_RTs = gt_RTs[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))

#             cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
#             cls_pred_bboxes =  pred_bboxes[pred_class_ids==cls_id, :] if len(pred_class_ids) else np.zeros((0, 4))
#             cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
#             cls_pred_RTs = pred_RTs[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
#             cls_pred_scales = pred_scales[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))



#             iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales,
#                                                                                            cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
#                                                                                            iou_thres_list)
#             if len(iou_pred_indices):
#                 cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
#                 cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
#                 cls_pred_scores = cls_pred_scores[iou_pred_indices]
#                 cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]


#             iou_pred_matches_all[cls_id] = np.concatenate((iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
#             cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
#             iou_pred_scores_all[cls_id] = np.concatenate((iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
#             assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
#             iou_gt_matches_all[cls_id] = np.concatenate((iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

#             if use_matches_for_pose:
#                 thres_ind = list(iou_thres_list).index(iou_pose_thres)

#                 iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                
#                 cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
#                 cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
#                 cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
#                 cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4))


#                 iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
#                 cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
#                 cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
#                 cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)



#             RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_RTs, cls_gt_handle_visibility, 
#                                               cls_pred_class_ids, cls_pred_RTs,
#                                               synset_names)


#             pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(RT_overlaps, 
#                                                                                   cls_pred_class_ids, 
#                                                                                   cls_gt_class_ids, 
#                                                                                   degree_thres_list, 
#                                                                                   shift_thres_list)
            

#             pose_pred_matches_all[cls_id] = np.concatenate((pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1)
            
#             cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
#             pose_pred_scores_all[cls_id]  = np.concatenate((pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
#             assert pose_pred_scores_all[cls_id].shape[2] == pose_pred_matches_all[cls_id].shape[2], '{} vs. {}'.format(pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape)
#             pose_gt_matches_all[cls_id] = np.concatenate((pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1)

    
    
#     # draw iou 3d AP vs. iou thresholds
#     fig_iou = plt.figure()
#     ax_iou = plt.subplot(111)
#     plt.ylabel('AP')
#     plt.ylim((0, 1))
#     plt.xlabel('3D IoU thresholds')
#     iou_output_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.png'.format(iou_thres_list[0], iou_thres_list[-1]))
#     iou_dict_pkl_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.pkl'.format(iou_thres_list[0], iou_thres_list[-1]))

#     iou_dict = {}
#     iou_dict['thres_list'] = iou_thres_list
#     for cls_id in range(1, num_classes):
#         class_name = synset_names[cls_id]
#         print(class_name)
#         for s, iou_thres in enumerate(iou_thres_list):
#             iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
#                                                                    iou_pred_scores_all[cls_id][s, :],
#                                                                    iou_gt_matches_all[cls_id][s, :])    
#         ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)
        
#     iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
#     ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
#     ax_iou.legend()
#     fig_iou.savefig(iou_output_path)
#     plt.close(fig_iou)

#     iou_dict['aps'] = iou_3d_aps
#     with open(iou_dict_pkl_path, 'wb') as f:
#         cPickle.dump(iou_dict, f)
    

#     # draw pose AP vs. thresholds
#     if use_matches_for_pose:
#         prefix='Pose_Only_'
#     else:
#         prefix='Pose_Detection_'


#     pose_dict_pkl_path = os.path.join(log_dir, prefix+'AP_{}-{}degree_{}-{}cm.pkl'.format(degree_thres_list[0], degree_thres_list[-2], 
#                                                                                           shift_thres_list[0], shift_thres_list[-2]))
#     pose_dict = {}
#     pose_dict['degree_thres'] = degree_thres_list
#     pose_dict['shift_thres_list'] = shift_thres_list

#     for i, degree_thres in enumerate(degree_thres_list):                
#         for j, shift_thres in enumerate(shift_thres_list):
#             # print(i, j)
#             for cls_id in range(1, num_classes):
#                 cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
#                 cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
#                 cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

#                 pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all, 
#                                                                         cls_pose_pred_scores_all, 
#                                                                         cls_pose_gt_matches_all)

#             pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
    
#     pose_dict['aps'] = pose_aps
#     with open(pose_dict_pkl_path, 'wb') as f:
#         cPickle.dump(pose_dict, f)


#     for cls_id in range(1, num_classes):
#         class_name = synset_names[cls_id]
#         print(class_name)
#         # print(np.amin(aps[i, :, :]), np.amax(aps[i, :, :]))
    
#         #ap_image = cv2.resize(pose_aps[cls_id, :, :]*255, (320, 320), interpolation = cv2.INTER_LINEAR)
#         fig_iou = plt.figure()
#         ax_iou = plt.subplot(111)
#         plt.ylabel('Rotation thresholds/degree')
#         plt.ylim((degree_thres_list[0], degree_thres_list[-2]))
#         plt.xlabel('translation/cm')
#         plt.xlim((shift_thres_list[0], shift_thres_list[-2]))
#         plt.imshow(pose_aps[cls_id, :-1, :-1], cmap='jet', interpolation='bilinear')

#         output_path = os.path.join(log_dir, prefix+'AP_{}_{}-{}degree_{}-{}cm.png'.format(class_name, 
#                                                                                    degree_thres_list[0], degree_thres_list[-2], 
#                                                                                    shift_thres_list[0], shift_thres_list[-2]))
#         plt.colorbar()
#         plt.savefig(output_path)
#         plt.close(fig_iou)        
    
#     #ap_mean_image = cv2.resize(pose_aps[-1, :, :]*255, (320, 320), interpolation = cv2.INTER_LINEAR) 
    
#     fig_pose = plt.figure()
#     ax_pose = plt.subplot(111)
#     plt.ylabel('Rotation thresholds/degree')
#     plt.ylim((degree_thres_list[0], degree_thres_list[-2]))
#     plt.xlabel('translation/cm')
#     plt.xlim((shift_thres_list[0], shift_thres_list[-2]))
#     plt.imshow(pose_aps[-1, :-1, :-1], cmap='jet', interpolation='bilinear')
#     output_path = os.path.join(log_dir, prefix+'mAP_{}-{}degree_{}-{}cm.png'.format(degree_thres_list[0], degree_thres_list[-2], 
#                                                                              shift_thres_list[0], shift_thres_list[-2]))
#     plt.colorbar()
#     plt.savefig(output_path)
#     plt.close(fig_pose)

    
#     fig_rot = plt.figure()
#     ax_rot = plt.subplot(111)
#     plt.ylabel('AP')
#     plt.ylim((0, 1.05))
#     plt.xlabel('translation/cm')
#     for cls_id in range(1, num_classes):
#         class_name = synset_names[cls_id]
#         print(class_name)
#         ax_rot.plot(shift_thres_list[:-1], pose_aps[cls_id, -1, :-1], label=class_name)
    
#     ax_rot.plot(shift_thres_list[:-1], pose_aps[-1, -1, :-1], label='mean')
#     output_path = os.path.join(log_dir, prefix+'mAP_{}-{}cm.png'.format(shift_thres_list[0], shift_thres_list[-2]))
#     ax_rot.legend()
#     fig_rot.savefig(output_path)
#     plt.close(fig_rot)

#     fig_trans = plt.figure()
#     ax_trans = plt.subplot(111)
#     plt.ylabel('AP')
#     plt.ylim((0, 1.05))

#     plt.xlabel('Rotation/degree')
#     for cls_id in range(1, num_classes):
#         class_name = synset_names[cls_id]
#         print(class_name)
#         ax_trans.plot(degree_thres_list[:-1], pose_aps[cls_id, :-1, -1], label=class_name)

#     ax_trans.plot(degree_thres_list[:-1], pose_aps[-1, :-1, -1], label='mean')
#     output_path = os.path.join(log_dir, prefix+'mAP_{}-{}degree.png'.format(degree_thres_list[0], degree_thres_list[-2]))
    
#     ax_trans.legend()
#     fig_trans.savefig(output_path)
#     plt.close(fig_trans)

#     iou_aps = iou_3d_aps
#     print('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.25)] * 100))
#     print('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.5)] * 100))
#     print('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(5)] * 100))
#     print('5 degree, 100cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(100)] * 100))
#     print('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(5)] * 100))
#     print('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(10)] * 100))
#     print('15 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(5)] * 100))
#     print('15 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(10)] * 100))


#     return iou_3d_aps, pose_aps