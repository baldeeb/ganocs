import torch
import torch.nn.functional as F
    # delta = (im - mu).pow(2).mean(-1, keepdim=True)
    # std = im.std(-1, keepdim=True)
    # return (std - target_std).abs().mean(-1)

def color_density(image, target=None):
    '''
    Args:
        image: (B, C, N, H, W)
        std: (3)
    Returns:
        loss: (N)'''
    im = image.flatten(-2, -1)
    if target is None:
        mu   = im.mean(-1)                    # (B, C, N)
        n_mu = mu.mean(-2, keepdim=True)      # (B, C, 1)
        loss = torch.norm(mu - n_mu, dim=-1)  # (B, C)
        return loss.mean()                    # (1)
    else:
        t  = target.flatten(-2, -1)
        t = F.one_hot(t, num_classes=im.shape[2])
        t = t.transpose(-1, -2).float()       # (B, C, N, HxW)    

        mu   = im.mean(-1)                    # (B, C, N)
        t_mu = t.mean(-1)                     # (B, C, N)
        loss = torch.norm(mu - t_mu, dim=-1)  # (B, C)
        return loss.mean()                    # (1)
