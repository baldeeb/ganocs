import pathlib as pl
import os
import torch
from models.nocs import get_nocs_resnet50_fpn
import logging

def save_model(model, path, retain_n=None):
    path = pl.Path(path)
    if not path.parent.exists():
        os.makedirs(path.parent)
    torch.save(model.state_dict(), path)
    logging.debug(f"Saved model to: {path}")
    if retain_n: 
        chkpts = sorted(path.parent.glob('*.pth'))
        chkpts = chkpts[-retain_n:]
        for c in chkpts: os.remove(c)
            
        
def load_nocs(checkpoint, **kwargs):
    # TODO: move this to a utils function
    path = pl.Path(checkpoint)
    if not path.is_file() or path.suffixes[-1] != '.pth':
        raise ValueError(f'Checkpoint {checkpoint} is not a valid path to a .pth file.')
    m = get_nocs_resnet50_fpn(**kwargs)
    m.load_state_dict(torch.load(checkpoint))
    return m