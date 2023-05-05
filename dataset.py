from torch.utils.data import (DataLoader, 
                              WeightedRandomSampler)
import pathlib as pl
import numpy as np
import json
import cv2

def stratify_values(values, num_bins=100):
    min_loss, max_loss = min(values), max(values)
    bins = np.linspace(min_loss, max_loss, num_bins)
    def _idxs_in_bin(i):
        return np.where((values >= bins[i]) & 
                        (values < bins[i+1]))[0]
    binned_idxs = [_idxs_in_bin(i) for i in range(num_bins-1)]
    binned_idxs = [b for b in binned_idxs if b.shape[0] > 0]
    binned_idxs = np.array(binned_idxs, dtype=np.object)
    bin_n = [b.shape[0] for b in binned_idxs]
    
    N = sum(bin_n)
    weights = [np.ones_like(idxs) * (N / b) 
               for idxs, b in zip(binned_idxs, bin_n)]
    idx_weights = np.concatenate(weights)
    idxs = np.concatenate(binned_idxs)
    return idxs, idx_weights

class NocsLossDataset():
    def __init__(self, directory):
        self._dir = pl.Path(directory)
        self._meta = self._dir / 'losses.json'
        files_n_losses = json.load(self._meta.open('r'))
        self._files = list(files_n_losses.keys())
        self._losses = np.array(list(files_n_losses.values()))
        self._losses = self._losses.astype(np.float32)
        self._idxs = list(range(len(self._losses)))

    def get_sample_weights(self, num_bins=100):
        self._idxs, weights = stratify_values(self._losses, 
                                              num_bins)
        return weights

    def __len__(self): return len(self._idxs)
    
    def __getitem__(self, i):
        file_dir = pl.Path(self._files[self._idxs[i]])
        img = cv2.imread(str(self._dir/file_dir), cv2.COLOR_BGR2RGB)
        img = img.transpose([2,0,1]).astype(np.float32) / 255.0
        return img, self._losses[self._idxs[i]]
    
    @staticmethod
    def get_dataloader(data_dir, 
                    batch_size=64, 
                    sampling=None):
        dataset = NocsLossDataset(data_dir)
        if sampling == 'stratified':
            weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(weights, 
                                            len(weights))
        else: sampler = None
        return DataLoader(dataset, 
                          sampler=sampler, 
                          batch_size=batch_size)