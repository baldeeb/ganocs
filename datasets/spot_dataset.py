import pathlib as pl
import numpy as np
import torch
import json
import cv2

class SpotDataset:
    '''A very simple dataset handler that expects three the directory to contain:
        meta.json, color/, depth/, & transforms/. Every file should start with a number except 
        for meta.json. The transforms and meta json files will contain a dict of source-pose 
        pairs. '''
    def __init__(self, directory, extension='png'):
        self._dir = pl.Path(directory)
        assert self._dir.is_dir(), f"Invalid directory: {directory}"
        self._files = {
            'color':     sorted(list(self._dir.glob(f'color/*.{extension}'))),
            'depth':     sorted(list(self._dir.glob(f'depth/*.{extension}'))),
            'transform': sorted(list(self._dir.glob(f'transforms/*.json'  ))),
        }
        self._meta = self._dir.glob('meta.json')

    def __len__(self):
        return len(self._files['color'])

    def __getitem__(self, i):

        with open(self._files['transform'][i]) as f:
            n = '_'.join(self._files['color'][i].stem.split('_')[1:])
            transform = json.load(f)[n]
        # with open(self._files['color'][i]) as f:
        color = cv2.imread(str(self._files['color'][i]), cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(self._files['depth'][i]), cv2.IMREAD_ANYDEPTH)
        return {
            'color':      torch.tensor(color.astype(np.float32))/255,
            'depth':      torch.tensor(depth.astype(np.float32))/1000,
            'camera_pose':  torch.tensor(transform, dtype=torch.float32),
            'intrinsics': self.intrinsic(),
        }

    def intrinsic(self):
        #TODO: Fix this
        # return np.array(self._meta['camera_intrinsic'])
        return torch.tensor([[552.02910122,   0.,           320.,], 
                             [0.,             552.02910122, 240.,], 
                             [0.,             0.,           1.,]])


def collate_fn(batch):
    rgb = torch.stack([v['color'].permute(2,0,1) for v in batch])
    for b in batch: b['boxes'] = torch.empty([0, 4])
    for b in batch: b['nocs'] = torch.empty([3, rgb.shape[-2], rgb.shape[-1]])
    return rgb, batch