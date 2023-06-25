# Pytorch Based NOCS

This is a re-implementation of the NOCS paper [link](TODO) over pytorch's [maskrcnn](TODO) implementation.

The model is verified on a dataset collected using [shapenet](TODO) and the [Habitat](TODO) simulator

**As of April 30 a few pieces are stil pending. Check the list below.**

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
    - [ ] box fitting (Umeyama optimization)
      - The umeyama implementation from the original nocs work is copied into utils/aligning.py. All that remains is using those to derive the box and pose metrics.s
  
## Code TODOs

- [ ] move model/nocs_uitl to uitls/ folder.
- [ ] move utils/nocs_detection_wrapper.py to model/
- [ ] rename nocs_roi_heads to roi_heads_with_nocs
- [ ] multiview loss and nocs alignment contains many acceptable failure cases.
        Create a custom Exception to identify acceptable failures. Currently
        all throw RuntimeWarnings.

## References

- **Pytorch example code:** pytorch has an example of DCGAN coded and explained 
[here](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

- **Diagnosing GAN training:** [This article](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/) 
goes over different failure modes and discusses the reason they occure and how to mitigate them.
