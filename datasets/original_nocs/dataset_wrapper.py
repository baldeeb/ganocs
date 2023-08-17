import torch
from .utils import (extract_bboxes, minimize_mask,)
import numpy as np
import pathlib as pl

############################################################
#  Data Formatting
############################################################
def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    # return images.astype(np.float32) - config.MEAN_PIXEL
    return images.astype(np.float32) / 255.0


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    # return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
    return (normalized_images * 255.0).astype(np.uint8)

############################################################
#  Data Generator
############################################################
# TODO: move into the dataset class!!!
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config, augment=False):
        """A generator that returns images and corresponding target class ids,
            bounding box deltas, and masks.

            dataset: The Dataset object to pick data from
            config: The model config object
            shuffle: If True, shuffles the samples before every epoch
            augment: If True, applies image augmentation to images (currently only
                     horizontal flips are supported)

            Returns a Python generator. Upon calling next() on it, the
            generator returns two lists, inputs and outputs. The containtes
            of the lists differs depending on the received arguments:
            inputs list:
            - images: [batch, H, W, C]
            - image_metas: [batch, size of image meta]
            REMOVED - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            REMOVED - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                        are those of the image unless use_mini_mask is True, in which
                        case they are defined in MINI_MASK_SHAPE.

            outputs list: Usually empty in regular training. But if detection_targets
                is True then the outputs list contains target class_ids, bbox deltas,
                and masks.
            """
        self.b = 0  # batch item index
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.dataset = dataset
        self.config = config
        self.augment = augment


    def __getitem__(self, image_index):
        # Get GT bounding boxes and masks for image.
        image_id = self.image_ids[image_index]

        image, depth, image_metas, gt_boxes, gt_masks, gt_coords, gt_domain_label, scale= \
            load_image_gt(self.dataset, self.config, image_id, augment=self.augment,
                          use_mini_mask=self.config.USE_MINI_MASK)
        
        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if np.sum(gt_boxes) <= 0:
            gt_class_ids = 0
 
        else:
            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            # rpn_match = rpn_match[:, np.newaxis]
            image = mold_image(image.astype(np.float32), self.config)
            gt_class_ids = gt_boxes[:,-1]

        return (image, 
                depth, 
                self.dataset.intrinsics,  
                # image_metas,
                gt_class_ids,
                gt_boxes, 
                gt_masks, 
                gt_coords, 
                gt_domain_label,
                scale)
    def __len__(self):
        return self.image_ids.shape[0]
   
def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False,load_scale = True):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """

    # Load image and mask
    if augment and dataset.subset == 'train':
        image, mask, coord, class_ids, scales, domain_label = dataset.load_augment_data(image_id)
    else:
        image = dataset.load_image(image_id)
        mask, coord, class_ids, scales, domain_label = dataset.load_mask(image_id)
    
    depth = dataset.load_depth(image_id)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    # Add class_id as the last value in bbox
    bbox = np.hstack((bbox, class_ids[:, np.newaxis]))

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        coord =  minimize_mask(bbox, coord, config.MINI_MASK_SHAPE)

    # Used to have:
    #   [image_id] +            # size=1
    #   list(image_shape) +     # size=3
    #   list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
    #   list(active_class_ids)  # size=num_classes  # some datasets dont have all calsses active
    image_meta = None

    if not load_scale: scales = None

    return (image, depth, image_meta, bbox, mask, coord, domain_label, scales)

