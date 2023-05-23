import pathlib as pl
import os
import torch
from torch_nocs.models.nocs import get_nocs_resnet50_fpn

def save_nocs(model, path):
    path = pl.Path(path)
    if not path.parent.exists():
        os.makedirs(path.parent)
    torch.save(model.state_dict(), path)

def load_nocs(checkpoint, **kwargs):
    # TODO: move this to a utils function
    path = pl.Path(checkpoint)
    if not path.is_file() or path.suffixes[-1] != '.pth':
        raise ValueError(f'Checkpoint {checkpoint} is not a valid path to a .pth file.')
    m = get_nocs_resnet50_fpn(**kwargs)
    m.load_state_dict(torch.load(checkpoint))
    return m