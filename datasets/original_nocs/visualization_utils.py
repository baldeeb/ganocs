'''These are tools for validating the data being loaded.

A data sample contains the following:
    - image, 
    - depth, 
    - self._dataset.intrinsics,  
    - gt_class_ids,
    - gt_boxes, 
    - gt_masks, 
    - gt_coords, 
    - gt_domain_label,
    - scale
    '''

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_2d_boxes_with_labels(ax, image, masks, boxes, labels, scores=None):
    '''Draws 2d boxes on image.
    
    Args:
        image: numpy array of shape (H, W, 3)
        boxes: numpy array of shape (N, 4)
        labels: numpy array of shape (N,)
        scores: numpy array of shape (N,)
    '''
        
    # Add overlayed masks on image
    for i, mask in enumerate(masks):
        mask = mask.astype(np.float32) * 1.0
        # mask = mask.astype(np.uint8) * 255
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
        image = cv2.addWeighted(image, 1, mask, 0.5, 0)

    # Draw boxes and labels
    ax.imshow(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label = labels[i]
        if scores is not None:
            score = scores[i]
            label = f'{label} - {score:.2f}'
        ax.text(x1, y1, label, color='r', fontsize=8)
    return ax

def visualize_original_nocs_data_point(axs, image, depth, masks, boxes, labels, nocs):
    draw_2d_boxes_with_labels(axs[0], image, masks, boxes, labels)
    axs[1].imshow(depth)
    axs[2].imshow(nocs.transpose(1,2,0))