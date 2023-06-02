from torch import (stack, 
                   meshgrid, 
                   arange, 
                   nonzero, 
                   inverse, 
                   cat, 
                   norm, 
                   ones_like, 
                   flatten,
                   FloatTensor)
from numpy import random

# def multiview_consistencry_loss(nocs, targets, n_pairs=None):
#     '''
#     Assuming that the batch samples are close to each other.
#     Args:
#         nocs:    (B, 3, N, H, W) where N is the number of bins
#         depth:   (B, 1,    H, W)
#         poses:   (B, N, 4, 4)
#         intrinsic:  (3, 3)
#     '''
#     B = nocs.shape[0]
#     mgrid = stack(meshgrid(arange(B), arange(B)), dim=-1)  # [B, B, 2]
#     idx_pairs = flatten(mgrid, 0, 1)  # [BxB, 2]
#     sample_idxs = arange(B*B)
#     if n_pairs is not None and n_pairs < B*B:
#         sample_idxs = random.choice(sample_idxs, n_pairs)
#     loss = []
#     for si in sample_idxs:
#         i1, i2 = idx_pairs[si]
#         if i1 == i2: continue
#         loss.append(
#             viewpair_consistency_loss(nocs[i1], targets[i1]['depth'], targets[i1]['pose'],
#                                       nocs[i2], targets[i2]['depth'], targets[i2]['pose'], 
#                                       targets[i1]['intrinsics']))
#     if len(loss) == 0: return 0
#     return sum(loss) / len(loss)


# def viewpair_consistency_loss(nocs1, depth1, pose1, nocs2, depth2, pose2, K):
#     '''
#     Args:
#         nocs{1,2}:  (3, N, H, W) where N is the number of bins
#         depth{1,2}: (      H, W)
#         pose{1,2}:  (4, 4)
#         intrinsic:  (3, 3)
#     '''
#     device = nocs1.device
#     depth1 = depth1.to(device)
#     depth2 = depth2.to(device)
#     pose1 = pose1.to(device)
#     pose2 = pose2.to(device)
#     K = K.float().to(device)

#     homogenized = lambda x : cat([x, ones_like(x[:, :1])], dim=1)

#     # Get transform from 1 to 2
#     twoTone = pose2 @ inverse(pose1)

#     # Select pixels with nocs map
#     ij1  = nonzero(nocs1.sum(dim=(0, 1)))
#     if ij1.shape[0] == 0: 
#         return 0
#     d1   = depth1[ij1[:, 0], ij1[:, 1]]
    
#     # Filter out points with no depth
#     valid = nonzero(d1 > 0).squeeze(1)
#     if valid.shape[0] == 0: 
#         return 0
#     ij1, d1  = ij1[valid], d1[valid]

#     # Project to 3D
#     uv1 = ij1[:, [1, 0]]
#     xyz1 = inverse(K) @ (homogenized(uv1) * d1[:, None]).T 

#     # Transform to second view
#     xyz2 = twoTone @ homogenized(xyz1.T).T

#     # Project to 2D of second view
#     uv2  = (K @ xyz2[:3] / xyz2[2])[:2].T.long()
#     ij2  = uv2[:, [1, 0]]

#     # Select only in-frame pixels
#     in_frame = nonzero((ij2[:, 0] >= 0) * (ij2[:, 1] >= 0) *
#                        (ij2[:, 0] < depth2.shape[0]) *
#                        (ij2[:, 1] < depth2.shape[1])).squeeze(1)
#     if in_frame.shape[0] == 0: 
#         return 0
#     ij1, ij2 = ij1[in_frame], ij2[in_frame]
#     xyz2 = xyz2[:, in_frame]

#     # Select only non-occluded samples
#     depth_eps = FloatTensor([0.02]).to(device)
#     d2 = depth2[ij2[:, 0], ij2[:, 1]]
#     visible_i = nonzero(((xyz2[2] - d2).abs() < depth_eps) * 
#                         (d2 > 0)).squeeze(1)
#     if visible_i.shape[0] == 0: 
#         return 0
#     ij1, ij2 = ij1[visible_i], ij2[visible_i]

#     # Compute the distance between associated pixels
#     n1 = nocs1[:, :, ij1[:, 0], ij1[:, 1]]
#     n2 = nocs2[:, :, ij2[:, 0], ij2[:, 1]]
#     loss = norm(n1 - n2, dim=1).mean()

#     # TODO: remove - temp ################################################
#     import numpy as np
#     def to_img(x):
#         x = ((x.argmax(dim=1) / x.shape[1]) * 255)
#         return x.clone().detach().cpu().long().numpy().transpose(1, 2, 0).astype(np.uint8)
    
#     k1 = [cv2.KeyPoint(int(p[0]), int(p[1]), size=1) for p in ij1]
#     k2 = [cv2.KeyPoint(int(p[0]), int(p[1]), size=1) for p in ij2]
#     img3 = cv2.drawMatches(to_img(nocs1), k1, 
#                            to_img(nocs2), k2, 
#                         [],#    np.stack([np.arange(len(k1)), np.arange(len(k1))]).astype(np.uint32).T,
#                            None, 
#                         #    matchColor = (0,255,0), # draw matches in green color
#                         #    singlePointColor = None,
#                         #    flags = 2
#                            )
#     cv2.imwrite("testing.jpg", img3)
#     #######################################################################

#     return loss






def multiview_consistency_loss(results, targets, scores, image_shapes, debugging=False):
    from utils.detection import NocsDetection
    detections = []
    for result, target, score in zip(results, targets, scores):
        if len(score) == 0: continue
        detections.append( NocsDetection(result['nocs'], result['boxes'], score,
                                         target['depth'], target['intrinsics'],
                                         image_shapes, target['pose']) )
    if debugging:
        # Visualize detections
        import cv2
        for i, d in enumerate(detections):
            cv2.imwrite(f'./data/temp/nocs{i}_in_head.png', d.get_as_image())

    B = len(detections)
    n_pairs = 100
    mgrid = stack(meshgrid(arange(B), arange(B)), dim=-1)  # [B, B, 2]
    idx_pairs = flatten(mgrid, 0, 1)  # [BxB, 2]
    sample_idxs = arange(B*B)
    if n_pairs is not None and n_pairs < B*B:
        sample_idxs = random.choice(sample_idxs, n_pairs)
    loss = []
    for si in sample_idxs:
        i1, i2 = idx_pairs[si]
        if i1 == i2: continue

        try:
            det1, det2 = detections[i1], detections[i2]
            ij1, ij2 = det1.get_associations(det2)
            n1, n2 = det1.get_nocs(ij1), det2.get_nocs(ij2)

            if debugging:
                cv2.imwrite("./data/temp/associations_visualization.jpg", 
                            det1.visualize_associations(det2, n_samples=10))

            loss.append(norm(n1 - n2, dim=1).mean())
        except:
            continue
    
    if len(loss) == 0: return 0
    return sum(loss) / len(loss)
