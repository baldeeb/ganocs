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
            'multiheaded_nocs': True,
        }

    def __init__(self, 
                 checkpoint, 
                 intrinsic, 
                 return_annotated_image=False, 
                 **kwargs):
        kwargs.update(self._default_config)
        self._device = kwargs['device'] if 'device' in kwargs \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_nocs(checkpoint, **kwargs)
        self.model.eval().to(self._device)
        self._K = self._to_ndarr(intrinsic)
        self._draw_box = return_annotated_image

    def _to_ndarr(self, a):
        if isinstance(a, torch.Tensor):
            return a.clone().detach().cpu().numpy()
        elif isinstance(a, list):
            return np.array(a)
        else: return a

    def __call__(self, images, depth):
        images = images.to(self._device)
        results = self.model(images)

        for i, (r, d) in enumerate(zip(results, depth)):
            results[i]['transforms'], results[i]['scales'] = [], []
            for m, n in zip(r['masks'], r['nocs']):
                Ts, Ss, _ = align(self._to_ndarr(m) > 0.5, 
                                  self._to_ndarr(n), 
                                self._to_ndarr(d), self._K)
                results[i]['transforms'].append(Ts)
                results[i]['scales'].append(Ss)
    
            if self._draw_box:
                annotated_img = self._to_ndarr(images[i].permute(1,2,0))
                for t, s, in zip(Ts, Ss):
                    annotated_img = draw_3d_boxes(annotated_img, 
                                                self._to_ndarr(t), 
                                                self._to_ndarr(s), 
                                                self._K)
                results[i]['annotated_image'] = annotated_img

        return results
            
