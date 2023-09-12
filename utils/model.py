from torch import nn
from typing import Iterator 

def get_model_parameters(model: nn.Module, recurse: bool = True, keys: list = None) -> Iterator[nn.Parameter]:
    for n, p in model.named_parameters(recurse=recurse):
        if keys is None: yield p
        elif any([k in n for k in keys]): 
            yield p