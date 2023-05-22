import torch 
import pathlib as pl

from utils.visualization import draw_3d_boxes
from utils.align import align
from utils.load_save import load_nocs

class NocsDetector:

    def __init__(self, checkpoint, intrinsic, device=None, return_annotated_image=False):
        self.model = load_nocs(checkpoint)
        self.model.eval()
        self._K = self._to_ndarr(intrinsic)
        self._device = device if device is not None \
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._draw_box = return_annotated_image

    def _to_ndarr(self, a):
        if isinstance(a, torch.Tensor):
            return a.clone().detach().cpu().numpy()
        else: return a

    def __call__(self, images, depth):
        images = images.to(self._device)
        results = self.model(images)

        output = {'transforms': [], 'scales': []}
        if self._draw_box: output['annotated_images'] = []
        for i, (m, n, d) in enumerate(zip(results['masks'], results['nocs'], depth)):
            Ts, Ss, _ = align(self._to_ndarr(m), 
                              self._to_ndarr(n), 
                              self._to_ndarr(d), 
                              self._K)
            output['transforms'].append(Ts)
            output['scales'].append(Ss)
    
            if self._draw_box:
                annotated_img = self._to_ndarr(images[i].permute(1,2,0))
                for t, s, in zip(Ts, Ss):
                    annotated_img = draw_3d_boxes(annotated_img, 
                                                self._to_ndarr(t), 
                                                self._to_ndarr(s), 
                                                self._K)
                output['annotated_images'].append(annotated_img)

        return output
            
