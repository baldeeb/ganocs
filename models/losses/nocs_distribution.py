import torch

    # delta = (im - mu).pow(2).mean(-1, keepdim=True)
    # std = im.std(-1, keepdim=True)
    # return (std - target_std).abs().mean(-1)

def color_density(image):
    '''
    Args:
        image: (B, C, N, H, W)
        std: (3)
    Returns:
        loss: (N)'''
    im = image.flatten(-2, -1)
    mu = im.mean(-1)                      # (B, C, N)
    n_mu = mu.mean(-2, keepdim=True)      # (B, C, 1)
    loss = torch.norm(mu - n_mu, dim=-1)  # (B, C)
    return loss.mean()                    # (1)
