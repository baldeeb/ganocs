from torch import (stack, 
                   meshgrid, 
                   arange,
                   norm,  
                   flatten,)
from numpy import random


def multiview_consistency_loss(results, targets, scores, image_shapes, n_pairs=100, debugging=True):
    from utils.detection import NocsDetection
    detections = []
    for result, target, score, shape in zip(results, targets, scores, image_shapes):
        if len(score) == 0: continue
        detections.append( NocsDetection(result['nocs'], result['boxes'], score,
                                         target['depth'], target['intrinsics'],
                                         shape, target['pose']) )
    if debugging:
        # Visualize detections
        import cv2
        for i, d in enumerate(detections):
            cv2.imwrite(f'./data/temp/nocs{i}_in_head.png', d.get_as_image())

    B = len(detections)
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
                import cv2
                cv2.imwrite("./data/temp/associations_visualization.jpg", 
                            det1.visualize_associations(det2, n_samples=10))

            loss.append(norm(n1 - n2, dim=1).mean())
        except:
            continue
    
    if len(loss) == 0: return 0
    return sum(loss) / len(loss)
