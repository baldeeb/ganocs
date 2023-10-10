
from torchvision.models.detection.transform import (GeneralizedRCNNTransform,  
                                                    paste_masks_in_image,
                                                    ImageList,
                                                    resize_boxes)
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
                targets[i]['nocs'] = self._resize(targets[i]['nocs'])
                if 'depth' in targets[i]:
                    H, W = targets[i]['depth'].shape[:2]
                    targets[i]['depth'] = self._resize(targets[i]['depth'].view(1, H, W))
        return images, targets

    def _resize(self, image: Tensor):
        image, _ = GeneralizedRCNNTransform.resize(self, image, None)
        return image


    def full_postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        # TODO: Fix... hacky...
        # Reason: postprocess relies on the training state.
        #   We want to run postprocess irrespective of that flag.
        '''
        This is expected to run after postprocess and 
        to support cached results. 

        The results are expected to contain a key 'nocs'
        that is a torch.Tensor of shape [B, C, N, H, W]
        where B is batch, C is number predictions, and N 
        is the number of bins.
        '''
        bin_index = 1        
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, 
                                                     image_shapes, 
                                                     original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "nocs" in pred:
                num_bins =  pred["nocs"]["x"].shape[bin_index]
                def process_nocs_dim(v):
                    v = v.argmax(bin_index) / num_bins 
                    return paste_masks_in_image(v.unsqueeze(bin_index), 
                                             pred["boxes"], o_im_s)
                nocs = cat([process_nocs_dim(pred['nocs']["x"]),
                            process_nocs_dim(pred['nocs']["y"]),
                            process_nocs_dim(pred['nocs']["z"])],
                            dim=1)
                result[i]["nocs"] = nocs
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = GeneralizedRCNNTransform.resize_boxes(
                                     self, keypoints,im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result

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
                    if num_bins > 1:  # Assume that it is classification and not regression.
                        v = v.argmax(bin_index) / num_bins 
                    v = paste_masks_in_image(v.unsqueeze(bin_index), 
                                             pred["boxes"], o_im_s)
                    return v
                nocs = cat([process_nocs_dim(pred["nocs"]["x"]),
                            process_nocs_dim(pred["nocs"]["y"]),
                            process_nocs_dim(pred["nocs"]["z"])],
                            dim=1)

                result[i]["nocs"] = nocs
        return result

