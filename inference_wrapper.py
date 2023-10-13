import torch 
import torchvision
import numpy as np

from utils.visualization import draw_3d_boxes
from utils.align import align
from utils.load_save import load_nocs

class NocsDetector:

    @property
    def _default_config(self):
        return {
            'maskrcnn_weights': torchvision.models.detection.mask_rcnn.MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            'nocs_loss': torch.nn.functional.cross_entropy,
            'nocs_num_bins': 32,
            'nocs_loss_mode': 'classification',
            'head_per_channel': True,
        }

    def __init__(self, 
                 checkpoint, 
                 intrinsic, 
                 return_annotated_image=False,
                 perform_alignment=True,
                 score_threshold=0.8,
                 **kwargs):
        # kwargs.update(self._default_config)
        args = self._default_config
        args.update(kwargs)
         
        self._device = args['device'] if 'device' in args \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_nocs(checkpoint, **args)
        self.model.eval().to(self._device)
        self._K = self._to_ndarr(intrinsic)
        self._draw_box = return_annotated_image
        self._align = perform_alignment
        self.score_threshold = score_threshold

    def _to_ndarr(self, a):
        if isinstance(a, torch.Tensor):
            return a.clone().detach().cpu().numpy()
        elif isinstance(a, list):
            return np.array(a)
        else: return a

    def _to_tensor(self, a):
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a).to(self._device)
        elif isinstance(a, torch.Tensor):
            a = a.to(self._device)
        if len(a.shape) == 3:
            a = a.unsqueeze(0)
        return a


    def __call__(self, images, depth):
        '''
        Args:
            images (torch.Tensor): [(B,) 3, H, W] a batch of (or single) images.
            depth (torch.Tensor): [(B,) H, W] a batch of (or single) depth maps.
        '''
            

        images = self._to_tensor(images)
        results = self.model(images)

        if len(depth.shape) == 2: depth = depth[None]

        for i, (r, d) in enumerate(zip(results, depth)):

            # prune out any results with low certainty
            selected_results = results[i]['scores'] > self.score_threshold
            for k in results[i].keys(): 
                results[i][k] = results[i][k][selected_results]

            results[i]['transforms'], results[i]['scales'] = [], []
            
            if self._align:
                annotated_img = self._to_ndarr(images[i].permute(1,2,0))
                for m, n in zip(r['masks'], r['nocs']):
                    Ts, Ss, _ = align(self._to_ndarr(m) > 0.5, 
                                    self._to_ndarr(n), 
                                    self._to_ndarr(d), self._K)
                    results[i]['transforms'].append(Ts)
                    results[i]['scales'].append(Ss)
        
                    if self._draw_box:
                        annotated_img = draw_3d_boxes(annotated_img, 
                                                    self._to_ndarr(Ts[0]), 
                                                    self._to_ndarr(Ss[0]), 
                                                    self._K)
                results[i]['annotated_image'] = annotated_img.astype(np.uint8)

        return results
            
