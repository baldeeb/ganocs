import os
import cv2
import json
import numpy as np
from time import time

class SaveCachedNocsAndLoss():
    def __init__(self, out_folder):
        if os.path.isdir(out_folder): 
            raise ValueError(f'{out_folder} already exists.') 
        else: os.makedirs(out_folder)
        self._folder = out_folder
        self._losses = {}
    def __call__(self, epoch, cache):
        for i, result in enumerate(cache):
            for j, nocs in enumerate(result['nocs']):
                f_name = f'{epoch}_{time()}_nocs.png'
                img = nocs.permute(1,2,0).detach().cpu().numpy()
                cv2.imwrite(f'{self._folder}/{f_name}', 
                            (img*255).astype(np.uint8))

                self._losses[f_name] = result['loss_nocs'][j].numpy().tolist()
    
    def save_losses(self):
        f = open(f'{self._folder}/losses.json', 'w+')
        json.dump(self._losses, f)
        f.close()