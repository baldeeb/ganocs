"""
Cropped and edited from util.py of the original NOCS repo.

previously noted:
    Mask R-CNN
    Common utility functions and classes.
    Copyright (c) 2017 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Waleed Abdulla
"""

import cv2
import numpy as np
from utils.evaluation.tools import get_3d_bbox, transform_coordinates_3d

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)
    return projected_coordinates


def draw(img, imgpts, axes, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = img.copy()

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)
    
    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)

    # draw axes
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3) ## y last
    return img


def draw_3d_boxes(image, transform, scale, intrinsic,
                  color=(255, 0, 0)):
    '''
    Args:
        image [H, W, 3] image to be drawn on
        transforms [4, 4] of the box that is to be drawn
        scale (float) of the box
    Returns:
        updated image of shape [H, W, 3]
    '''

    xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    transformed_axes = transform_coordinates_3d(xyz_axis, transform)
    projected_axes = calculate_2d_projections(transformed_axes, intrinsic)

    bbox_3d = get_3d_bbox(scale, 0)
    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, transform)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsic)
    return draw(image, projected_bbox, projected_axes, color)



