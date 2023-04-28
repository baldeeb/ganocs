from torch import cat, arange
def select_labels(nocs_dict, labels):
    '''
    Args:
        data (torch.Tensor): of shape [B, L, ...] where B
            is the batch size L is the labels one of which
            we need to retain.
        labels (List[int]): containing the supported labels
        matches (List[int]): of shape [B] where each value
            refers to the index of the labels vector. The
            label at that index will be retained for each
            batch element.
    '''
    l = cat(labels) - 1
    ids = arange(l.size(0), device=l.device)
    discard_ids = arange(l.size(0), device=l.device)[l==-1]
    l[l==-1] = 0
    for _, (k, v) in enumerate(nocs_dict.items()):
        v = v[ids, l]
        v[discard_ids] = v[discard_ids] * 0
        nocs_dict[k] = v
    return nocs_dict