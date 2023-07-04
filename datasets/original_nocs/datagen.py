import torch
from .utils import (resize_image, 
                    resize_mask, 
                    extract_bboxes, 
                    generate_pyramid_anchors, 
                    minimize_mask, 
                    compute_iou,
                    )
import numpy as np

############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
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

        # # Anchors
        # # [anchor_count, (y1, x1, y2, x2)]
        # self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
        #                                          config.RPN_ANCHOR_RATIOS,
        #                                          config.BACKBONE_SHAPES,
        #                                          config.BACKBONE_STRIDES,
        #                                          config.RPN_ANCHOR_STRIDE)

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
            # rpn_bbox = 0
            # rpn_match = 0
            gt_class_ids = 0
 
        else:
            # RPN Targets
            # rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors, gt_boxes, self.config)

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

            # # Convert
            # images = torch.from_numpy(images.transpose(2, 0, 1)).float()
            # image_metas = torch.from_numpy(image_metas)
            # # rpn_match = torch.from_numpy(rpn_match)
            # # rpn_bbox = torch.from_numpy(rpn_bbox).float()
            # gt_class_ids = torch.from_numpy(gt_class_ids)
            # gt_boxes = torch.from_numpy(gt_boxes).float()
            # gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()
            # gt_coords = torch.from_numpy(gt_coords)
            # # gt_domain_label = torch.from_numpy(gt_domain_label)

        return (image, 
                depth, 
                self.dataset.intrinsics,  
                # image_metas,
                # rpn_match, rpn_bbox, 
                gt_class_ids,
                gt_boxes, 
                gt_masks, 
                gt_coords, 
                gt_domain_label)
    def __len__(self):
        return self.image_ids.shape[0]
    

def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False,load_scale = False):
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

    shape = image.shape
    image, window, scale, padding = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = resize_mask(mask, scale, padding)
    coord = resize_mask(coord, scale, padding)


    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    # Add class_id as the last value in bbox
    bbox = np.hstack((bbox, class_ids[:, np.newaxis]))

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    # active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    # source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    # active_class_ids[source_class_ids] = 1

    active_class_ids = np.ones([dataset.num_classes], dtype=np.int32)

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        coord =  minimize_mask(bbox, coord, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)

    if not load_scale: scales = None

    return (image, depth, image_meta, bbox, mask, coord, domain_label, scales)

# def build_rpn_targets(image_shape, anchors, gt_boxes, config):
#     """Given the anchors and GT boxes, compute overlaps and identify positive
#     anchors and deltas to refine them to match their corresponding GT boxes.

#     anchors: [num_anchors, (y1, x1, y2, x2)]
#     gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, class_id)]

#     Returns:
#     rpn_match: [N] (int32) matches between anchors and GT boxes.
#                1 = positive anchor, -1 = negative anchor, 0 = neutral
#     rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
#     """
#     # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
#     rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
#     # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
#     rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

#     # Areas of anchors and GT boxes
#     gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
#     anchor_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])

#     # Compute overlaps [num_anchors, num_gt_boxes]
#     # Each cell contains the IoU of an anchor and GT box.
#     overlaps = np.zeros((anchors.shape[0], gt_boxes.shape[0]))
#     for i in range(overlaps.shape[1]):
#         gt = gt_boxes[i][:4]
#         overlaps[:,i] = compute_iou(gt, anchors, gt_box_area[i], anchor_area)

#     # Match anchors to GT Boxes
#     # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
#     # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
#     # Neutral anchors are those that don't match the conditions above, 
#     # and they don't influence the loss function.
#     # However, don't keep any GT box unmatched (rare, but happens). Instead,
#     # match it to the closest anchor (even if its max IoU is < 0.3).
#     #
#     # 1. Set negative anchors first. It gets overwritten if a gt box is matched to them.
#     anchor_iou_argmax = np.argmax(overlaps, axis=1)
#     anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
#     rpn_match[anchor_iou_max < 0.3] = -1
#     # 2. Set an anchor for each GT box (regardless of IoU value).
#     # TODO: If multiple anchors have the same IoU match all of them
#     gt_iou_argmax = np.argmax(overlaps, axis=0)
#     rpn_match[gt_iou_argmax] = 1
#     # 3. Set anchors with high overlap as positive.
#     rpn_match[anchor_iou_max >= 0.7] = 1

#     # Subsample to balance positive and negative anchors
#     # Don't let positives be more than half the anchors
#     ids = np.where(rpn_match == 1)[0]
#     extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
#     if extra > 0:
#         # Reset the extra ones to neutral
#         ids = np.random.choice(ids, extra, replace=False)
#         rpn_match[ids] = 0
#     # Same for negative proposals
#     ids = np.where(rpn_match == -1)[0]
#     extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(rpn_match == 1))
#     if extra > 0:
#         # Rest the extra ones to neutral
#         ids = np.random.choice(ids, extra, replace=False)
#         rpn_match[ids] = 0

#     # For positive anchors, compute shift and scale needed to transform them
#     # to match the corresponding GT boxes.
#     ids = np.where(rpn_match == 1)[0]
#     ix = 0  # index into rpn_bbox
#     # TODO: use box_refinment() rather that duplicating the code here
#     for i, a in zip(ids, anchors[ids]):
#         # Closest gt box (it might have IoU < 0.7)
#         gt = gt_boxes[anchor_iou_argmax[i], :4]

#         # Convert coordinates to center plus width/height.
#         # GT Box
#         gt_h = gt[2] - gt[0]
#         gt_w = gt[3] - gt[1]
#         gt_center_y = gt[0] + 0.5 * gt_h
#         gt_center_x = gt[1] + 0.5 * gt_w
#         # Anchor
#         a_h = a[2] - a[0]
#         a_w = a[3] - a[1]
#         a_center_y = a[0] + 0.5 * a_h
#         a_center_x = a[1] + 0.5 * a_w

#         # Compute the bbox refinement that the RPN should predict.
#         rpn_bbox[ix] = [
#             (gt_center_y - a_center_y) / a_h,
#             (gt_center_x - a_center_x) / a_w,
#             np.log(gt_h / a_h),
#             np.log(gt_w / a_w),
#         ]
#         # Normalize
#         rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
#         ix += 1

#     return rpn_match, rpn_bbox
