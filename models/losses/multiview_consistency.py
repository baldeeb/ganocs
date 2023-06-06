from torch import (stack, 
                   meshgrid, 
                   arange,
                   norm,  
                   flatten,
                   randperm)
from numpy import random
from utils.detection import NocsDetection
import cv2  # For debugging

def _sample_n_pairs(n, B):
    '''Returns pairs of indices of size n from B choices.
    Chooses with no repetition and without ignoring self-pairs.'''
    mgrid = stack(meshgrid(arange(B), arange(B)), dim=-1)  # [B, B, 2]
    pairs = flatten(mgrid, 0, 1)  # [BxB, 2]
    paird_idxs = arange(B*B)
    if n is not None and n < B*B:
        paird_idxs = random.choice(paird_idxs, n)
    return pairs[paird_idxs]

def multiview_consistency_loss(results, targets, scores, image_shapes, 
                               n_pairs=100, debugging=False):
    detections = []
    for result, target, score, shape in zip(results, targets, scores, image_shapes):
        if len(score) == 0: continue
        detections.append( NocsDetection(result['nocs'], result['boxes'], score,
                                         target['depth'], target['intrinsics'],
                                         shape, target['pose']) )
    if debugging:  # Visualize detections
        for i, d in enumerate(detections):
            cv2.imwrite(f'./data/temp/nocs{i}_in_head.png', d.get_as_image())

    pairs = _sample_n_pairs(n_pairs, len(detections))
    loss = []
    for i1, i2 in pairs:
        if i1 == i2: continue
        try:
            det1, det2 = detections[i1], detections[i2]
            ij1, ij2 = det1.get_associations(det2)
            n1, n2 = det1.get_nocs(ij1), det2.get_nocs(ij2)

            neg1, neg2 = randperm(len(n1)), randperm(len(n2))

            pos_loss = norm(n1 - n2, dim=1).mean()
            neg_loss = (  norm(n1[neg1] - n2, dim=1) 
                        + norm(n1- n2[neg2],  dim=1) ).mean()

            if debugging:
                cv2.imwrite("./data/temp/associations_visualization.jpg", 
                            det1.visualize_associations(det2, n_samples=10))
            loss.append( pos_loss / neg_loss )
        except:
            continue
    
    if len(loss) == 0: return 0
    return sum(loss) / len(loss)
