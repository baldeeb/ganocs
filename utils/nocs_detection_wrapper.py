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
                   where, 
                   FloatTensor)
from models.nocs_util import paste_in_image
from torchvision import transforms as T
from utils.misc import get_ijs
from utils.byoc_utils.alignment import test_align
class NocsDetection:
    def __init__(self, nocs_pred, boxs, scores, masks, labels, depth, intrinsics, img_shape, camera_pose=None):
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
        assert len(scores) > 0, 'Empty detections are not allowed.'
        
        self.shape = img_shape
        self.device = boxs.device
        
        self.preds = nocs_pred
        self.nocs = NocsDetection.nocs_in_box(nocs_pred, boxs) # list(Num_dets x [Channels, Bins, H, W]])
        self.boxs = boxs.to(self.device)
        self.scores = scores.to(self.device)
        self.camera_pose = camera_pose.to(self.device) if camera_pose is not None else None
        self.masks = masks.to(self.device)
        self.labels = labels.to(self.device)
        self.occludsion_eps = FloatTensor([0.02]).to(self.device)
        self.box_mask = self.mask_of_nocs_boxes()

        resize = T.Resize(size=img_shape)
        self.depth = resize(depth[None])[0].to(self.device)

        self.K = intrinsics
        dshape = depth.shape[-2:]
        self.K[0] *= img_shape[0] / dshape[0]
        self.K[1] *= img_shape[1] / dshape[1]
        self.K = self.K.to(self.device).float()

    def __len__(self):
        return len(self.scores)

    def masks_in_image(self, idxs=None, threshold=0.5):
        if idxs is not None:
            m, b = self.masks[[idxs]], self.boxs[[idxs]]  
        else: 
            m, b = self.masks, self.boxs
        masks_im = paste_in_image(m, b, self.shape[0], self.shape[1])
        return masks_im.sum(0) > threshold


    def mask_of_nocs_boxes(self):
        ''' Returns a mask of shape [H, W] where each pixel is 1 if
        the pixel is in any of the boxes, and 0 otherwise.
        '''
        H, W = self.shape
        mask = zeros(H, W).bool().to(self.device)
        for box in self.boxs:
            mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
        return mask


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
    

    # def mask_of_nocs_preds(self):
    #  # Seemed lie a less efficient mask_of_nocs_boxes
    #     H, W = self.shape
    #     mask = ones(H, W, device=self.device).bool()
    #     for k in self.pred.keys():
    #         m = paste_in_image(self.pred[k], self.box, H, W)
    #         m = m.sum((0,1)) == 0
    #         mask = mask * m
    #     return ~mask


    def have_valid_data(self, ij):
        ''' A pixel is said to be worthy if nocs map is not zero 
        at that pixel, depth is not zero, and the pixel is in the 
        image frame.
        Args:
            ij: (N, 2)
        Returns:
            has_data: (N) bool indicating whether an index has valid data.
        Raises:
            RuntimeWarning: If no valid pixels are found.'''
        nocs_mask  = self.box_mask
        depth_mask = self.depth > 0
        mask = nocs_mask * depth_mask 
        has_data = mask[ij[:, 0], ij[:, 1]]
        if has_data.sum() == 0: 
            raise RuntimeWarning("has_valid_data: found no data where queried.")
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
    

    def get_depth_reprojection(self, ij=None, idx=None):
        '''Returns: 
            ij: [N, 2] pixels of the image that were successfully projected.
            xyz: [N, 3] projections of valid depth found.'''
        if idx is not None: 
            ij = self.masks_in_image(idx).squeeze(0).nonzero()
            ij = ij[self.have_valid_data(ij)]
        elif ij is None: 
            ij = get_ijs(*self.depth.shape[-2:], 
                        device=self.device)
            ij = ij[self.have_valid_data(ij)]
        # Select those with valid data
        d = self.depth[ij[:, 0], ij[:, 1]]
        # Project to 3D
        uv = ij[:, [1, 0]]
        xyz = inverse(self.K) @ (self._homogenized(uv) * d[:, None]).T
        return ij, xyz.T
        

    def get_nocs_reprojection(self, ij=None, idx=None):
        '''Returns: nocs: (N, 2)'''
        if idx is not None: 
            ij = self.masks_in_image(idx).squeeze(0).nonzero()
            ij = ij[self.have_valid_data(ij)]
        elif ij is None: 
            ij = get_ijs(*self.depth.shape[-2:], 
                        device=self.device)
            ij = ij[self.have_valid_data(ij)]
            
        # Select those with valid data
        nocs = self.get_nocs(ij)

        def soft_argmax(x):
            v = torch.arange(x.shape[2], device=x.device).float()
            return (v[None, None, :] * x.softmax(dim=2)).sum(dim=2)

        nocs = (soft_argmax(nocs) / nocs.shape[2]) - 0.5
        return nocs


    def get_associations(self, other:'NocsDetection'):
        
        assert self.camera_pose is not None and other.camera_pose is not None, \
            "Poses must be set before calling get_associations_with"

        # Get 3D points in self frame
        ij, xyz = self.get_depth_reprojection()

        # Transform to second view
        otherTself = (other.camera_pose @ inverse(self.camera_pose)).float()
        other_xyz = otherTself @ self._homogenized(xyz).T

        # Project to 2D of second view
        other_uv  = (other.K @ other_xyz[:3] / other_xyz[2])[:2].T.long()
        other_ij  = other_uv[:, [1, 0]]

        # Select in other frame
        in_other_frame = other.in_frame(other_ij)
        if in_other_frame.sum() == 0: 
            raise RuntimeWarning("get_associations: frame did not overlap.")
        ij, other_ij = ij[in_other_frame], other_ij[in_other_frame]
        other_xyz = other_xyz[:, in_other_frame]

        # Select valid projects
        valid_proj = other.have_valid_data(other_ij)
        if valid_proj.sum() == 0: 
            raise RuntimeWarning("get_associations: has no valid depth or NOCS in other view.")
        ij, other_ij = ij[valid_proj], other_ij[valid_proj]
        other_xyz = other_xyz[:, valid_proj]

        # Select only non-occluded samples
        is_visible = ~other.occluded(other_ij, other_xyz[2])
        if is_visible.sum() == 0: 
            raise RuntimeWarning("get_associations: All projections are occluded in other view.")
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
            nocs: (N, 3, C) where C is the number of bins.
        '''
        nocs_channels, nocs_bins = self.nocs[0].shape[0:2]
        result = zeros(ij.shape[0], nocs_channels, nocs_bins).to(self.device)
        
        # Get sorted indices of self.scores
        sorted_i = self.scores.argsort(descending=False)

        # Get the box with the highest score that contains ij
        for i in sorted_i:
            box = self.boxs[i]
            in_box = (ij[:, 0] >= box[1]) * (ij[:, 1] >= box[0]) * \
                     (ij[:, 0] <= box[3]) * (ij[:, 1] <= box[2])
            if in_box.sum() == 0: continue
            found_ij = (ij[in_box] - box[:2][None, [1,0]]).long()
            result[in_box] = self.nocs[i][:, :, found_ij[:, 0], found_ij[:, 1]].permute(2, 0, 1)

        return result


    def get_as_image(self, score_threshold=0.4, as_torch=False):
        selected_idxs = self.scores > score_threshold
        stacked = stack(list(self.preds.values()), dim=1)[selected_idxs]
        select_boxes = self.boxs[selected_idxs]
        im = zeros(*stacked.shape[1:3], self.shape[0], self.shape[1], device=self.device)
        
        for ci in range(stacked.shape[1]):
            pred_c = stacked[:, ci]
            reshaped = paste_in_image(pred_c, select_boxes, self.shape[0], self.shape[1],)
            im[ci] = reshaped.sum(0)

        if as_torch: return im

        def numpify(x):
            import numpy as np
            x = ((x.argmax(dim=1) / x.shape[1]) * 255)
            return x.clone().detach().cpu().long().numpy().transpose(1, 2, 0).astype(np.uint8)
        
        return numpify(im)


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
    

    def get_reprojections(self, idx):
        ij = self.masks_in_image(idx).squeeze(0).nonzero()
        ij = ij[self.have_valid_data(ij)]
        n = self.get_nocs_reprojection(  ij = ij )
        _, d = self.get_depth_reprojection( ij = ij )
        return n, d

    def align_nocs_to_depth(self, depth_Rt=None):
        '''
        Args:
            depth_Rt: (4, 4) transformation matrix to apply to depth points
                before alignment.
            Returns:
                Rts: (N, 4, 4) transformation matrices of object poses'''
        Rts, losses, dists, weights = [], [], [], []
        for i in range(self.__len__()):
            try: 
                nocs_xyz, depth_xyz = self.get_reprojections( idx = i )
                
                if depth_Rt is not None: # Transform depth to different view.
                    depth_xyz = self._homogenize_dim_one(depth_xyz.clone().T)
                    depth_xyz = (depth_Rt @ depth_xyz)[:3].T
                
                Rt, loss, dist = test_align( nocs_xyz[None,:, :], 
                                             depth_xyz[None,:, :] )
                Rts.append(Rt.squeeze(0))
                losses.append(loss)
                dists.append(dist)
                weights.append(self.scores[i])
            except Exception as e:
                print(f'align_nocs_to_depth: {e}'); continue
        if len(Rts) == 0: return None, torch.zeros(1), torch.zeros(1)
        return stack(Rts), stack(losses), stack(weights)[:, None]  # dists?
    

    @staticmethod
    def _homogenize_dim_one(pts):
        '''
        Args:
            pts [N, 3, ...]: list of points with the second dimention to homogenize
        '''
        bottom = torch.zeros_like(pts[:1], device=pts.device)
        bottom[0] = 1.0
        return torch.concat([pts, bottom], dim=0)


    def object_pose_consistency(self, other:'NocsDetection'):
        '''After aligning nocs to depth and derivin object poses, 
        this function finds associated objects in self and other then
        determines the pose consistency between them.'''
        dt_thresh = 0.3  # TODO: make configurable
        
        # Get relative camera pose from self to other
        selfTother = (other.camera_pose @ inverse(self.camera_pose)).float()

        # Align nocs to depth
        Rts1, aloss1, _ = self.align_nocs_to_depth(selfTother)
        Rts2, aloss2, _ = other.align_nocs_to_depth()
        
        # Get average alignment loss
        avg_alignment_loss = ( aloss1.mean() + aloss2.mean() ) / 2

        # If no objects are found, return
        if Rts1 is None: 
            # return 0, avg_alignment_loss
            raise RuntimeWarning("No objects found in self image." +\
                                " Failed to associate objects.")
        if Rts2 is None:
            # return 0, avg_alignment_loss
            raise RuntimeWarning("No objects found in other images." +\
                                " Failed to associate objects.")

        # Get associations based on translatal proximity
        t1, t2 = Rts1[:, :3, 3], Rts2[:, :3, 3]
        dist_mat = (t1[:, None] - t2[None]).norm(dim=2)
        
        # Select the pairs with the closest centers
        t1i, t2i = dist_mat.argmin(dim=0), torch.arange(len(t2), device=self.device)
        select = (dist_mat[t1i, t2i] < dt_thresh)
        t1i, t2i = t1i[select], t2i[select]

        # Get the associated Rts
        Rt1, Rt2 = Rts1[t1i], Rts2[t2i]
        if len(Rts1) == 0 or len(Rts2) == 0: 
            # return 0, avg_alignment_loss
            raise RuntimeWarning("No detected objects have close centers." +\
                                " Failed to associate objects.")
        

        # # TEST ###################################################################
        # _, P = self.get_depth_reprojection()
        # _, Q = other.get_depth_reprojection()

        # P = torch.concat([P.T, torch.ones_like(P.T[:1], device=self.device)], dim=0)
        # P_hat = (twoTone @ P).T

        # import matplotlib.pyplot as plt
        # lim = 5
        # ax = plt.figure().add_subplot(projection='3d')
        # def scatter(v,l,m='o', c='b'): 
        #     v = v.clone().detach().cpu().numpy()
        #     ax.scatter(v[:, 0], v[:, 1], v[:, 2], marker=m, label=l, color=c)
        # scatter(Q, 'Q', 'o', 'r')
        # scatter(P_hat, 'P hat', 'X', 'b')
        # scatter(P, 'original P', 'X', 'g')
        # ax.legend()
        # for set_lim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]: set_lim(-lim, lim)
        # ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        # ax.view_init(elev=20., azim=-35, roll=0)
        # plt.show()
        # ##########################################################################

        dist = ( Rt1 - Rt2 ).norm(dim=(1,2))
        

        if len(dist) == 0:
            raise RuntimeWarning("Failed to associate objects.")
        return dist.mean(), avg_alignment_loss