# TODO: maybe remove?

from torch import (arange, 
                   stack,
                   meshgrid,
                   flatten,)

def get_ijs(H, W, device='cuda'):
    ah, aw = arange(H, device=device), arange(W, device=device)
    return flatten(stack(meshgrid(ah, aw), dim=-1), 0, 1)