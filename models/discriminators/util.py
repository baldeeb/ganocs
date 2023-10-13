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

