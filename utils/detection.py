import torch.nn.functional as F
import torch
from torch import (stack, 
                   meshgrid, 
                   arange, 
                   inverse, 
                   cat, 
                   zeros,
                   ones,
                   ones_like, 
                   flatten,
                   FloatTensor)
from models.nocs_util import paste_in_image
from torchvision import transforms as T

class NocsDetection:
    def __init__(self, nocs_pred, box, score, depth, intrinsics, img_shape, pose=None):
        '''
        Houses a single detection.
        Args:
            nocs_pred (torch.Tensor): of shape [C, N, H, W]
            box (torch.Tensor): of shape [4]
            score (torch.Tensor): of shape [1]
            depth (torch.Tensor): of shape [H, W]
            intrinsics (torch.Tensor): of shape [3, 3]
            img_shape (torch.Tensor): of shape [2] (H, W) assues all detections are from same shaped images
            '''
        assert len(img_shape) == 2, 'This class always assumes the source are images of the same shape.'
        assert len(score) > 0, 'Empty detections are not allowed.'
        self.device = box.device
        self.pred = nocs_pred
        self.nocs = NocsDetection.nocs_in_box(nocs_pred, box) # list(Num_dets x [Channels, Bins, H, W]])
        self.box = box.to(self.device)
        self.score = score.to(self.device)
        self.pose = pose.to(self.device) if pose is not None else None
        self.occludsion_eps = FloatTensor([0.02]).to(self.device)
        self.shape = img_shape

        dshape = depth.shape[-2:]
        resize = T.Resize(size=img_shape)
        self.depth = resize(depth[None])[0].to(self.device)

        self.K = intrinsics
        self.K[0] *= img_shape[0] / dshape[0]
        self.K[1] *= img_shape[1] / dshape[1]
        self.K = self.K.to(self.device).float()


    @staticmethod 
    def nocs_in_box(pred, box):
        w = (box[:, 2] - box[:, 0] + 1).long().clamp(1)
        h = (box[:, 3] - box[:, 1] + 1).long().clamp(1)
        if isinstance(pred, dict):
            nocs = stack(list(pred.values()), dim=1)
        else: nocs = pred
        return [F.interpolate(nocs[i], size=(h[i], w[i]), 
                              mode="bilinear", 
                              align_corners=False) 
                for i in range(len(nocs))]
    

    def mask_of_nocs_preds(self):
        H, W = self.shape
        mask = ones(H, W).bool().to(self.depth.device)
        for k in self.pred.keys():
            m = paste_in_image(self.pred[k], self.box, H, W)
            # import matplotlib.pyplot as plt
            # plt.imshow(m[0].clone().detach().cpu().numpy().sum(0))
            # plt.show()
            m = m.sum((0,1)) != 0
            mask = mask * m.to(mask)
        # for n, b in zip(self.nocs, self.box):
            # m = paste_in_image(n, b, H, W).sum(0, 1) > 0
            # masks.append(m)
        return mask


    def have_valid_data(self, ij):
        ''' A pixel is said to be worthy if nocs map is not zero 
        at that pixel, depth is not zero, and the pixel is in the 
        image frame.
        Args:
            ij: (N, 2)
        '''
        # nocs_mask  = self.nocs.sum(dim=(0, 1))>0
        nocs_mask  = self.mask_of_nocs_preds()
        depth_mask = self.depth > 0
        mask = nocs_mask * depth_mask 
        has_data = mask[ij[:, 0], ij[:, 1]]
        return has_data


    def in_frame(self, ij):
        H, W = self.depth.shape[-2:]
        return ((ij[:, 0] >= 0) * (ij[:, 1] >= 0) *
               (ij[:, 0] < H) * (ij[:, 1] < W))


    def occluded(self, ij, z):
        d = self.depth[ij[:, 0], ij[:, 1]]
        return (z - d) > self.occludsion_eps


    def _homogenized(self, x): 
        return cat([x, ones_like(x[:, :1])], dim=1)
    

    def get_associations(self, other:'NocsDetection'):
        
        assert self.pose is not None and other.pose is not None, \
            "Poses must be set before calling get_associations_with"

        H, W = self.depth.shape[-2:]
        ij = flatten(stack(meshgrid(arange(H), arange(W)), dim=-1), 0, 1)
        ij = ij.to(self.device)

        # Select those with valid data
        of_valid_data = self.have_valid_data(ij)
        if of_valid_data.sum() == 0: raise ValueError("No valid pixels.")
        ij = ij[of_valid_data]
        d = self.depth[ij[:, 0], ij[:, 1]]

        # Get transform from 1 to 2
        otherTself = (other.pose @ inverse(self.pose)).float()

        # Project to 3D
        uv = ij[:, [1, 0]]
        xyz = inverse(self.K) @ (self._homogenized(uv) * d[:, None]).T 

        # Transform to second view
        other_xyz = otherTself @ self._homogenized(xyz.T).T

        # Project to 2D of second view
        other_uv  = (other.K @ other_xyz[:3] / other_xyz[2])[:2].T.long()
        other_ij  = other_uv[:, [1, 0]]

        # Select in other frame
        in_other_frame = other.in_frame(other_ij)
        if in_other_frame.sum() == 0: raise ValueError("No points in other frame.")
        ij, other_ij = ij[in_other_frame], other_ij[in_other_frame]
        other_xyz = other_xyz[:, in_other_frame]

        # Select valid projects
        valid_proj = other.have_valid_data(other_ij)
        if valid_proj.sum() == 0: raise ValueError("No valid projections.")
        ij, other_ij = ij[valid_proj], other_ij[valid_proj]
        other_xyz = other_xyz[:, valid_proj]

        # Select only non-occluded samples
        is_visible = ~other.occluded(other_ij, other_xyz[2])
        if is_visible.sum() == 0: raise ValueError("No visible points.")
        ij, other_ij = ij[is_visible], other_ij[is_visible]

        return ij.long(), other_ij.long()
    

    def get_nocs(self, ij):
        ''' Returns the nocs at the given pixel indices.
        To avoid creating a full nocs map, this method first
        checks if a box contains ij, then finds the pixel relative
        to the box corner.

        Args:
            ij: (N, 2)
        Returns:
            nocs: (N, C) where C is the number of bins.
        '''
        nocs_channels, nocs_bins = self.nocs[0].shape[0:2]
        result = zeros(ij.shape[0], nocs_channels, nocs_bins).to(self.device)
        
        # Get sorted indices of self.scores
        sorted_i = self.score.argsort(descending=True)

        # Get the box with the highest score that contains ij
        for i in sorted_i:
            box = self.box[i]
            in_box = (ij[:, 0] >= box[1]) * (ij[:, 1] >= box[0]) * \
                     (ij[:, 0] <= box[3]) * (ij[:, 1] <= box[2])
            if in_box.sum() == 0: continue
            found_ij = (ij[in_box] - box[:2][None, [1,0]]).long()
            result[in_box] = self.nocs[i][:, :, found_ij[:, 0], found_ij[:, 1]].permute(2, 0, 1)

        return result


    def get_as_image(self):
        import numpy as np
        selected_idxs = self.score > 0.4
        stacked = stack(list(self.pred.values()), dim=1)[selected_idxs]
        select_boxes = self.box[selected_idxs]
        im = zeros(*stacked.shape[1:3], self.shape[0], self.shape[1])
        
        for ci in range(stacked.shape[1]):
            pred_c = stacked[:, ci]
            reshaped = paste_in_image(pred_c, select_boxes, self.shape[0], self.shape[1],)
            im[ci] = reshaped.sum(0)

        def to_img(x):
            x = ((x.argmax(dim=1) / x.shape[1]) * 255)
            return x.clone().detach().cpu().long().numpy().transpose(1, 2, 0).astype(np.uint8)
        
        return to_img(im)


    def visualize_associations(self, other, n_samples=5000):
        import cv2
        ij1, ij2 = self.get_associations(other)
        n_samples = min([len(ij1), len(ij2), n_samples])
        select = torch.randperm(len(ij1))[:n_samples]

        k1 = [cv2.KeyPoint(int(p[0]), int(p[1]), size=1) for p in ij1[select]]
        k2 = [cv2.KeyPoint(int(p[0]), int(p[1]), size=1) for p in ij2[select]]

        matches = [cv2.DMatch(i, i, 0, 0) for i in range(len(k1))]
        return cv2.drawMatches(self.get_as_image(), k1, 
                               other.get_as_image(), k2, 
                               matches, None, 
                               matchColor = (0,255,0),
                               singlePointColor = None,
                               flags = 2)