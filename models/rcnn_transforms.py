
from torchvision.models.detection.transform import (GeneralizedRCNNTransform,  
                                                    paste_masks_in_image,
                                                    ImageList)
from typing import (List, Optional, Tuple, Dict)
from torch import (Tensor, cat)

class GeneralizedRCNNTransformWithNocs(GeneralizedRCNNTransform):
    '''
    Expands GeneralizaleRCNNTransforms to include NOCS postprocessing.
    This class explicitly calls postprocessing as is called by RCNN
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        images, targets = GeneralizedRCNNTransform.forward(self, images, targets)
        if targets is not None: 
            for i in range(len(targets)):
                targets[i]['nocs'] = self.resize_nocs(targets[i]['nocs'])
        return images, targets

    def resize_nocs(self, nocs: Tensor):
        nocs, _ = GeneralizedRCNNTransform.resize(self, nocs, None)
        return nocs

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        '''
        The results are expected to contain a key 'nocs'
        that is a torch.Tensor of shape [B, C, N, H, W]
        where B is batch, C is number predictions, and N 
        is the number of bins.
        '''
        result = GeneralizedRCNNTransform.postprocess(self, 
                                                      result, 
                                                      image_shapes, 
                                                      original_image_sizes)
        if self.training: return result

        bin_index = 1        
        for i, (pred, o_im_s) in enumerate(zip(result, original_image_sizes)):
            if "nocs" in pred:
                num_bins =  pred["nocs"]["x"].shape[bin_index]
                def process_nocs_dim(v):
                    v = v.argmax(bin_index) / num_bins 
                    v = paste_masks_in_image(v.unsqueeze(bin_index), pred["boxes"], o_im_s)

                    # return v.softmax(0).argmax(0) * 255 / num_bins
                    return v
                nocs = cat([process_nocs_dim(pred["nocs"]["x"]),
                            process_nocs_dim(pred["nocs"]["y"]),
                            process_nocs_dim(pred["nocs"]["z"])],
                            dim=1)

                result[i]["nocs"] = nocs
        return result

