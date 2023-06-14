from torch import (stack, 
                   meshgrid, 
                   arange,
                   norm,  
                   flatten,
                   randperm)
from numpy import random
from utils.nocs_detection_wrapper import NocsDetection
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
                               n_pairs=100, debugging=False,
                               mode='alignment' # 'pixelwise' or 'alignment'
                               ):
    detections = []
    for result, target, score, shape in zip(results, targets, scores, image_shapes):
        if len(score) == 0: continue
        detections.append( NocsDetection(result['nocs'], result['boxes'], score,
                                         result['masks'], result['labels'],
                                         target['depth'], target['intrinsics'],
                                         shape, target['camera_pose'] # TODO: change to 'camera_pose'
                                        ) )
    if debugging:  # Visualize detections
        for i, d in enumerate(detections):
            cv2.imwrite(f'./data/temp/nocs{i}_in_head.png', d.get_as_image())

    try :
        if mode == 'pixelwise':
            loss = multiview_pixelwise_consistency_loss(detections, n_pairs, debugging)
        elif mode == 'alignment':
            loss = multiview_alignment_consistency_loss(detections)
        else:
            raise RuntimeError(f'Unknown multiview mode: {mode}')
        return loss
    except RuntimeWarning as e:
        print(e)  # TODO: log this in a better way
        return 0



def multiview_pixelwise_consistency_loss(detections, n_pairs=100, debugging=False):
    pairs = _sample_n_pairs(n_pairs, len(detections))
    loss = []
    for i1, i2 in pairs:
        if i1 == i2: continue
        try:
            det1, det2 = detections[i1], detections[i2]
            ij1, ij2 = det1.get_associations(det2)
            count = int(min([len(det1), len(det2)]) * 0.5)  # TODO: make parameter
            n1, n2 = det1.get_nocs(ij1)[:count], det2.get_nocs(ij2)[:count]

            neg1, neg2 = randperm(len(n1)), randperm(len(n2))

            pos_loss = norm(n1 - n2, dim=1).mean()
            neg_loss = (  norm(n1[neg1] - n2, dim=1) 
                        + norm(n1- n2[neg2],  dim=1) ).mean()

            if debugging:
                cv2.imwrite("./data/temp/associations_visualization.jpg", 
                            det1.visualize_associations(det2, n_samples=10))
            if pos_loss == neg_loss == 0: 
                raise RuntimeWarning("Pixelwise loss oddly perfect...Skipped.")
            loss.append( pos_loss / ( neg_loss + 1e-5) )
        except RuntimeWarning as e:
            print(e); continue
    
    if len(loss) == 0:
        raise RuntimeWarning("Pixelwise multiview loss: No image pairs yielded a loss...")
    return sum(loss) / len(loss)



def multiview_alignment_consistency_loss(detections, n_pairs=100):

    # Using consistency of object pose as loss
    pairs = _sample_n_pairs(n_pairs, len(detections))
    losses = {'alignment': [], 'object_pose': []}
    for i1, i2 in pairs:
        if i1 == i2: continue
        try:
            det1, det2 = detections[i1], detections[i2]
            obj_pose_loss, alignment_loss = det1.object_pose_consistency(det2)
            if not obj_pose_loss.isnan(): 
                losses['object_pose'].append(obj_pose_loss)
            # if not alignment_loss.isnan(): 
            #   losses['alignment'].append(alignment_loss)
        except RuntimeWarning as e: print(e); continue  # For the purpose of debugging.
            
    if all([len(l)==0 for l in losses.values()]): 
        raise RuntimeWarning("Object pose-consistency loss: No image pairs yielded a loss...")

    # Return the sum of the mean of each loss type    
    avgs = [sum(l)/ len(l) for l in losses.values() if len(l) > 0]
    return sum(avgs)








################################################################################################
################################  For Debugging ###############################################
################################################################################################

def display_optim_points(P, Q, lim=1):
    import matplotlib.pyplot as plt
    q = Q.clone().detach().cpu().numpy()
    p = P.clone().detach().cpu().numpy()
    ax = plt.figure().add_subplot(projection='3d')
    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x, y) points are plotted on the x and z axes.
    ax.scatter(q[:, 0], q[:, 1], q[:, 2], 'o', label='Q')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], label='P')
    # Make legend, set axes limits and labels
    ax.legend()
    for set_lim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]: set_lim(-lim, lim)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35, roll=0)
    plt.show()

def display_masked_nocs(det, i):
    import matplotlib.pyplot as plt
    plt.figure()
    mask = det.masks_in_image(i).clone().detach().cpu().numpy().squeeze()[:, :, None]
    nocs = det.get_as_image()
    plt.imshow( nocs * mask ); plt.show()

def show_depth(det):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(det.depth.clone().detach().squeeze().cpu().numpy())
    plt.show()