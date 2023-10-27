import torch

def get_multiple_discriminators(disc_type, count, *args, **kwargs):
    return [disc_type(*args, **kwargs) for _ in range(count)]

def accuracy(value:torch.Tensor, real:bool):
    value = value.flatten()
    if real:
        acc = torch.sum(value >= 0.5).item() / (len(value) + 1e-8)
    else:
        acc = torch.sum(value < 0.5).item() / (len(value) + 1e-8)
    return acc


def sample_balancer(a:torch.Tensor, b:torch.Tensor, dim:int=0):
    '''Given two batches of data, make sure they both
    have the same amount of samples by randomly sampling
    from the one with excess'''
    len_a, len_b  = a.shape[dim],  b.shape[dim]
    def select(x:torch.Tensor, n:int):
        return x.index_select(dim, torch.randperm(n))
    if len_a > len_b:
        a = select(a, b.shape[dim])
    elif len_a < len_b:
        b = select(b, a.shape[dim])
    return a, b