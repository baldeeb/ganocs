from omegaconf import DictConfig
from torch.utils.data import DataLoader
from .datagen import Dataset
import torch

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
        self._batch_size = batch_size
        self._iters = [iter(s) for s in self._sets]
        self._exhausted = torch.zeros(len(self._iters))

        if 'collate' in kwargs: self._collate = kwargs['collate']  
        else:                   self._collate = lambda x: x
    
    def _init_dataset(self, set_cfg:DictConfig):
        dataset = set_cfg.loader
        if set_cfg.type == 'real':
            dataset.load_real_scenes(set_cfg.dataset_dir)
        if set_cfg.type == 'synthetic':
            dataset.load_camera_scenes(set_cfg.dataset_dir)
        dataset.prepare(set_cfg.class_map)
        dataset = Dataset(dataset, set_cfg.loader.config, augment=set_cfg.augment)
        return DataLoader(dataset, 
                          shuffle=set_cfg.shuffle, 
                          batch_size=1,
                          collate_fn=lambda x: x)
    
    def __getitem__(self, _):
        '''NOTE: Not great but a hot fix to get things tested.'''
        set_idxs = torch.multinomial(self._set_weights, self._batch_size, replacement=True)
        out = []
        for set_i in set_idxs:
            try:
                # yield next(self._iters[set_i])
                # print(f'set index: {set_i}')
                out.append(next(self._iters[set_i])[0])
            except StopIteration:
                self._exhausted[set_i] = 1
                if all(self._exhausted): raise StopIteration
                self._iters[set_i] = iter(self._sets[set_i])
                out.append(next(self._iters[set_i])[0])
                # yield next(self._iters[set_i])
        if self._collate is not None: 
            return self._collate(out)
        return out

    def __len__(self):
        return sum([len(s) for s in self._sets])