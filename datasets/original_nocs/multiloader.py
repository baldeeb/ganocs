from omegaconf import DictConfig
from torch.utils.data import DataLoader
from .datagen import Dataset
from .dataset import BadDataException
import torch
import logging
import numpy as np

# NOTE: https://github.com/sahithchada/NOCS_PyTorch/blob/f4ed85efec3b39476bda74ef84a314fcbcf8f0b3/model.py#L1786
#     is a good reference summarizing how the dataset was used previously.

class NOCSMultiDatasetLoader:
    '''packages the data of the original NOCS dataset into a 
    dataloader the functions as expected by the rest of the code. 
    '''
    def __init__(self, datasets, batch_size, dataset_priorities, **kwargs):
        self._sets = []
        for set_cfg in datasets.values():
            self._sets.append(self._init_dataset(set_cfg)) 
        self._set_weights = torch.tensor(dataset_priorities)
        assert len(self._sets) == len(self._set_weights), 'dataset_priorities must have same length as datasets'
        self._batch_size = batch_size
        self._iters = [iter(s) for s in self._sets]
        self._exhausted = set()
        self._collate = kwargs.get('collate', None)
    
    def _init_dataset(self, set_cfg:DictConfig):
        dataset = set_cfg.loader
        if set_cfg.type == 'real':
            dataset.load_real_scenes(set_cfg.dataset_dir)
        if set_cfg.type == 'synthetic':
            dataset.load_camera_scenes(set_cfg.dataset_dir)
        dataset.prepare(set_cfg.class_map)
        dataset = Dataset(dataset, set_cfg.loader.config, augment=set_cfg.augment)
        # wrap in dataset to allow shuffling data.
        return DataLoader(dataset, 
                          shuffle=set_cfg.shuffle, 
                          batch_size=1,
                          collate_fn=lambda x: x)
    

    def __iter__(self):
        self._exhausted = set()
        return self

    def __next__(self):
        '''
        NOTE: Not great but a hot fix to get things tested.
        '''
        out = []
        while len(out) < self._batch_size:
            for set_i in torch.multinomial(self._set_weights, 1):
                try:
                    data = next(self._iters[set_i])[0]
                except StopIteration as e:
                    # TODO: should I just remove the exhausted dataset?
                    self._exhausted.add(set_i)
                    logging.warning(f'Exhausted dataset {set_i}\n{e}')
                    if all([i in self._exhausted for i in range(len(self._iters))]): 
                           raise StopIteration
                    self._iters[set_i] = iter(self._sets[set_i])
                    data = next(self._iters[set_i])[0]
                # These are ugly feature of the original dataset.
                except BadDataException as e:
                    logging.warning(f'Bad data in dataset {set_i}\n{e}')
                    continue
                if (isinstance(data[3], np.ndarray) and len(data[3]) == 0) \
                    or (isinstance(data[3], int) and data[3] == 0): 
                    continue  # ignore data with no labels
                out.append(data)
        if self._collate is not None: 
            return self._collate(out)
        return out

    def __len__(self):
        return sum([len(s) for s in self._sets])