
from torchvision.models.detection.transform import (GeneralizedRCNNTransform, 
                                                    resize_boxes, 
                                                    paste_masks_in_image)
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import torch 
from torch import Tensor

class GeneralizedRCNNTransformWithNocs(GeneralizedRCNNTransform):
    '''
    Expands GeneralizaleRCNNTransforms to include NOCS postprocessing.
    This class explicitly calls postprocessing as is called by RCNN
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        result = GeneralizedRCNNTransform.postprocess(self, 
                                                      result, 
                                                      image_shapes, 
                                                      original_image_sizes)
        if self.training: return result

        for i, (pred, o_im_s) in enumerate(zip(result, original_image_sizes)):
            if "nocs" in pred:
                num_bins =  pred["nocs"]["x"].shape[0]
                def process_nocs_dim(v):
                    v = v.softmax(0).argmax(1) / num_bins
                    v = paste_masks_in_image(v.unsqueeze(1), pred["boxes"], o_im_s)

                    # return v.softmax(0).argmax(0) * 255 / num_bins
                    return v
                nocs = torch.cat([process_nocs_dim(pred["nocs"]["x"]),
                                 process_nocs_dim(pred["nocs"]["y"]),
                                 process_nocs_dim(pred["nocs"]["z"])],
                                 dim=1)

                result[i]["nocs"] = nocs
        return result

