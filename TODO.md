## Additions

### Towards Loss learning

The code contains some added features for experiments that were ran.

- [x] results caching at train time.
- [x] cache saving.
- [x] loss prediction model.

**NOTE**: this track was abandoned in favor of using GANS

### Towards Learning 2D-3D correspondances with GANs

Initial proposed adding a discriminator to the nocs head training.

- [x] discriminator added
- [x] trained and tested

This track yielded poor outcomes and the conclusion was that:

- instead a full GAN model should be trained in parallel given image crops 
  as input rather than being given regions of interest.

## Not Implemented portions

    - [x] nocs head.
    - [ ] symmetry loss
      - [x] implemented nocs map rotation
      - [ ] pass in config that indicates what labels have what symmetries
      - [ ] integrate the symmetry loss by:
          - Rotating the ground-truth map.
          - Multiply the rotated maps with the mask to ensure that the segmentation is retained.
          - Get the loss against all rotated ground-truth maps.
          - apply the argmin over all losses against rotated maps.
    - [x] box fitting (Umeyama optimization)
      - The umeyama implementation from the original nocs work is copied into utils/aligning.py. All that remains is using those to derive the box and pose metrics.s
    - [ ] Augmentation for habitat data.
  
## Code TODOs

- [x] move model/nocs_uitl to uitls/ folder.
- [ ] move utils/nocs_detection_wrapper.py to model/
- [ ] rename nocs_roi_heads to roi_heads_with_nocs
- [ ] multiview loss and nocs alignment contains many acceptable failure cases.
        Create a custom Exception to identify acceptable failures. Currently
        all throw RuntimeWarnings.
