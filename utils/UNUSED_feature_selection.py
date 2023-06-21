'''Given the NOCS feature we would like to grasp 
    along with a nocs and depth maps, this file deals
    with selecting the closest feature to that we are
    interested in.'''

import numpy as np

def select_nocs_feature(feature, nocs, depth):
    '''
    Args:
        feature (np.ndaray): [3] elements corresponding to the 
            nocs RGB values we would like to grasp.
        depth (np.ndarray): [H, W] depth map.
        nocs (np.ndarray): [3, H, W] nocs map.'''
    
    # Nocs distance from freature
    nocs_dist = np.linalg.norm(nocs - feature[:, None, None], axis=0)

    # Arg sort nocs distance
    idxs = np.argsort(nocs_dist, axis=None)

    # Get indices of top 10% of closest pixels
    n = int(0.1 * nocs_dist.size)
    top_idxs = idxs[:n]
    
    # Get the depth points of top closest nocs pixels
    depth_points = depth.flatten()[top_idxs]

    # Cluster depth points
    clusters = cluster_depth_points(depth_points)

    # Select mean of largest cluster
    cluster_sizes = [len(c) for c in clusters]
    largest_cluster_idx = np.argmax(cluster_sizes)
    largest_cluster = clusters[largest_cluster_idx]
    mean_depth = np.mean(largest_cluster)


def cluster_depth_points(depth):
    '''Given a list of depth points, cluster them
        into groups based on a threshold.'''