import torch
from torch.utils.data import DataLoader
from .utils import (extract_bboxes, minimize_mask,)
import numpy as np
import pathlib as pl
import logging

class NOCSDataloader():
    def __init__(self, dataset, data_info, batch_size, class_map, **kwargs):
        
        self._dataset = dataset
        self._class_map = class_map
        self.shuffle=kwargs.get('shuffle', False)
        self._load_data(data_info, class_map)
        self._collate = kwargs.get('collate', None)
        self._batch_size = batch_size
        self._step_count = 0
        self._steps_per_epoch = kwargs.get('steps_per_epoch', None)
        self.augment = kwargs.get('augment', False)
        
    
    def _load_data(self, data_info, class_map):
        self._sources, self._source_weights = [], []
        for k, v in data_info.items(): 
            assert k not in self._sources, 'Same data source was added twice'
            self._sources.append(k)
            self._source_weights.append(v.weight)
            if k == 'Real':
                self._dataset.load_real_scenes(v.dataset_dir)
            elif k == 'CAMERA':
                self._dataset.load_camera_scenes(v.dataset_dir)
            elif k == 'coco':
                self._dataset.load_coco(v.dataset_dir, 
                                        self._dataset.subset, 
                                        class_map)
            elif k == 'HABITAT': 
                raise NotImplementedError('Habitat data is not yet set up.')
            self._dataset.prepare(class_map)
        self._source_weights = torch.tensor(self._source_weights).float()
        self._register_source_ids()

    def _register_source_ids(self): 
        self._sources_and_idxs, self._counters = {}, {}
        for k, v in self._dataset.source_image_ids.items():
            assert k in self._sources, f'Source {k} didn\'t properly registering!'
            if self.shuffle: np.random.shuffle(v)
            self._sources_and_idxs[k] = v
            self._counters[k] = 0

    def __iter__(self):
        self._register_source_ids()  # this will reset counters on dataset idxs
        self._step_count = 0
        return self
    
    def __len__(self):
        l = len(self._dataset) // self._batch_size
        if self._steps_per_epoch: 
            l = min(self._steps_per_epoch, l) 
        return l
    
    def __next__(self):
        out = []
        while len(out) < self._batch_size:
            if self._batch_exhausted(): raise StopIteration
            
            # sample source and image
            source_i = torch.multinomial(self._source_weights, 1).item()

            try:
                data = self._get_data(source_i)
                if self._no_visible_objects(data): continue
                self._step_count += 1
            except StopIteration as e:
                # TODO: should I just remove the exhausted dataset?
                logging.debug(f'Exhausted dataset {source_i}\n{e}')
                continue
            # # These are ugly feature of the original dataset.
            # except BadDataException as e:
            #     logging.debug(f'Bad data in dataset {set_i}\n{e}')
            #     continue
            out.append(data)
        if self._collate is not None: return self._collate(out)
        return out

    def _no_visible_objects(self, data):
        '''Returns true if there are no labels in data.'''
        return (isinstance(data[3], np.ndarray) and len(data[3]) == 0) \
                    or (isinstance(data[3], int) and data[3] == 0)

    def _batch_exhausted(self):
        if all([c == None for c in self._counters.values()]): 
            logging.debug('All datasets exhausted.')
            return True
        elif self._steps_per_epoch and self._step_count >= self._steps_per_epoch:
            logging.debug('An epoch worth of batches has been dispensed.')
            return True
        else:
            return False

    def _get_image_id(self, source_i):
        source = self._sources[source_i]
        idx = self._counters[source]
        if idx >= len(self._sources_and_idxs[source]):
            self._counters[source] = None
            raise StopIteration(f'Data source {source} is exhausted.')
        self._counters[source] += 1
        image_id = self._sources_and_idxs[source][idx]
        return image_id
    
    def _get_data(self, source_i):
        image_id = self._get_image_id(source_i)
        image, depth, image_metas, gt_boxes, gt_masks, gt_coords, gt_domain_label, scale= \
            load_image_gt(self._dataset, self._dataset.config, image_id, augment=self.augment)
        
        if np.sum(gt_boxes) <= 0:
            # skip data without labels; no interesting categories.
            gt_class_ids = 0
        else:
            image = image.astype(np.float32) / 255.0
            gt_class_ids = gt_boxes[:,-1]
            if gt_boxes.shape[0] > self._dataset.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self._dataset.config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

        return (image, 
                depth, 
                self._dataset.intrinsics,  
                # image_metas,
                gt_class_ids,
                gt_boxes, 
                gt_masks, 
                gt_coords, 
                gt_domain_label,
                scale)


def load_image_gt(dataset, config, image_id, augment=False):
    # Load image and mask
    if augment and dataset.subset == 'train':
        image, mask, coord, class_ids, scales, domain_label = dataset.load_augment_data(image_id)
    else:
        image = dataset.load_image(image_id)
        mask, coord, class_ids, scales, domain_label = dataset.load_mask(image_id)
    
    depth = dataset.load_depth(image_id)
    if depth is None: depth = np.zeros((image.shape[0], image.shape[1], 1))

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    # Add class_id as the last value in bbox
    bbox = np.hstack((bbox, class_ids[:, np.newaxis]))

    # Resize masks to smaller size to reduce memory usage
    if config.USE_MINI_MASK:
        mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        coord =  minimize_mask(bbox, coord, config.MINI_MASK_SHAPE)

    image_meta = None

    return (image, depth, image_meta, bbox, mask, coord, domain_label, scales)

