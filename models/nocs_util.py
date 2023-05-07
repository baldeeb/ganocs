from torch import cat, arange, Tensor, stack
def select_labels(nocs_dict, 
                  labels):
    '''
    Assumes labels are 1-indexes of the nocs_dict.
    Args:
        data (torch.Tensor): of shape [B, L, ...] where B
            is the batch size L is the labels one of which
            we need to retain.
        labels (List[int]): containing the supported labels
    '''
    l = cat(labels) - 1
    ids = arange(l.size(0), device=l.device)
    discard_ids = arange(l.size(0), device=l.device)[l==-1]
    l[l==-1] = 0
    for k, v in nocs_dict.items():
        v = v[ids, l]
        v[discard_ids] = v[discard_ids] * 0
        nocs_dict[k] = v
    return nocs_dict

def rotate_nocs(nocs:Tensor, 
                masks:Tensor, 
                rotation:Tensor, 
                times:int=1
            )->Tensor:
    '''
    Args:
        nocs [B, 3, H, W]
        mask [B, 1, H, W] of boolean dtype.
        rotation [*, 3, 3] where * can be 1 or B.
        times indicates the number of times to rotate
    Returns: 
        A rotated version of the nocs mask. If times==1
        the shape is retained, otherwise the returned 
        tensor will be of shape [B, times, 3, H, W].
    '''
    rots = [rotation]
    for _ in range(times-1):
        rots.append(rotation @ rots[-1])
    rots = stack(rots, dim=1)

    B, C, H, W = nocs.shape
    r =  rots @ nocs[:, None].flatten(-2,-1)
    r = r.reshape(B, times, C, H, W) * masks[:, None]
    return r.squeeze(1)

def separate_image_results(source, reference):
    '''splits the first dimention of source based on 
    the first 2 dimensions of the reference.'''
    _per_img = [b.shape[0] for b in reference]
    if isinstance(source, Tensor):
        return source.split(_per_img)
    elif isinstance(source, dict):
        for k, v in source.items(): 
            source[k] = v.split(_per_img)
        return source

def select_nocs_proposals(nocs_proposals, labels, num_classes):
    '''Selects the nocs proposals that match the labels. Reshapes the nocs proposals 
    to separate results from every image in the batch.
    Args:
        nocs_proposals (Dict[str,Tensor]): Each tensor is of shape [B, C, N, H, W]
            where B is the batch x predictions, C is the number of classes, N is the
            number of bins of nocs, H and W are the height and width of the image.
        labels (Tensor): Each tensor is of shape [B] indicating which class is
            correct for each prediction.
        num_classes (int): The number of classes that nocs can predict.
            Used as a precaution to avoid selecting labels that are out of bounds.
    Returns:
        A list of dictionaries where each dictionary contains the nocs proposals
        for a single image in the batch.
    '''
    # Select the predicted labels
    for i, l in enumerate(labels):
        labels[i][l>num_classes] = 0
    nocs_proposals = select_labels(nocs_proposals, 
                                    labels)
    # Split batch to batches
    nocs_proposals = separate_image_results(nocs_proposals, 
                                                    labels)
    return [{k:v[i] for k,v in nocs_proposals.items()} 
            for i in range(len(labels))]
