from omegaconf import DictConfig
from torch.utils.data import DataLoader
from .dataset_wrapper import Dataset
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
        self._set_cfgs = list(datasets.values())
        self._sets = [self._get_dataset(i) for i in range(len(self._set_cfgs))]
        self._loaders = self._iters = [None]*len(self._set_cfgs)
        for i in range(len(self._set_cfgs)): self._set_loader(i)
        self._exhausted = set()
        self._set_weights = torch.tensor(dataset_priorities)
        assert len(self._loaders) == len(self._set_weights), 'dataset_priorities must have same length as datasets'
        self._batch_size = batch_size
        self._step_count = 0
        self._collate = kwargs.get('collate', None)
        self._steps_per_epoch = kwargs.get('steps_per_epoch', None)
    

    @property
    def batch_size(self): return self._batch_size


    def _get_dataset(self, idx:int):
        set_cfg:DictConfig = self._set_cfgs[idx]
        dataset = set_cfg.loader
        if set_cfg.type == 'real':
            dataset.load_real_scenes(set_cfg.dataset_dir)
        if set_cfg.type == 'synthetic':
            dataset.load_camera_scenes(set_cfg.dataset_dir)
        dataset.prepare(set_cfg.class_map)
        return Dataset(dataset, set_cfg.loader.config, augment=set_cfg.augment)
    

    def _set_loader(self, idx:int):
        # wrap in dataset to allow shuffling data.
        dataset = self._sets[idx]
        cfg = self._set_cfgs[idx]
        self._loaders[idx] = DataLoader(dataset, 
                                        shuffle=cfg.shuffle, 
                                        batch_size=1,
                                        collate_fn=lambda x: x)
        self._iters[idx] = iter(self._loaders[idx])
    

    def __iter__(self):
        self._exhausted = set()
        self._step_count = 0
        return self


    def _batch_exhausted(self):
        if all([i in self._exhausted for i in range(len(self._iters))]): 
            logging.debug('All datasets exhausted.')
            return True
        elif self._steps_per_epoch and self._step_count >= self._steps_per_epoch:
            logging.debug('An epoch worth of batches has been dispensed.')
            return True
        else:
            return False


    def __next__(self):
        '''
        NOTE: Not great but a hot fix to get things tested.
        '''
        out = []
        while len(out) < self._batch_size:
            if self._batch_exhausted(): raise StopIteration
            set_i = torch.multinomial(self._set_weights, 1).item()
            try:
                data = next(self._iters[set_i])[0]
                if (isinstance(data[3], np.ndarray) and len(data[3]) == 0) \
                    or (isinstance(data[3], int) and data[3] == 0): 
                    continue  # ignore data with no labels
                self._step_count += 1
            except StopIteration as e:
                # TODO: should I just remove the exhausted dataset?
                logging.debug(f'Exhausted dataset {set_i}\n{e}')
                self._exhausted.add(set_i)
                self._set_loader(set_i)
                continue
            # These are ugly feature of the original dataset.
            except BadDataException as e:
                logging.debug(f'Bad data in dataset {set_i}\n{e}')
                continue
            out.append(data)
        if self._collate is not None: return self._collate(out)
        return out


    def __len__(self):
        total_len = sum([len(s) for s in self._loaders])
        if self._steps_per_epoch: 
            return min(self._steps_per_epoch, total_len) 
        else:
            return total_len