# Pytorch Based NOCS

This is a re-implementation of the NOCS paper [link](TODO) over pytorch's [maskrcnn](TODO) implementation.

The model is verified on a dataset collected using [shapenet](TODO) and the [Habitat](TODO) simulator

**As of April 30 a few pieces are stil pending. Check the list below.**

## Additions

The code contains some added features for experiments that were ran.

- [ ] results caching at train time.
- [ ] cache saving.
- [ ] loss prediction model.

## Unimplemented portions

- [x] nocs head
- [ ] symmetry loss
    - [x] implemented nocs map rotation
    - [ ] pass in config that indicates what labels have what symmetries
    - [ ] integrate the symmetry loss by: 
        - Rotating the ground-truth map.
        - Multiply the rotated maps with the mask to ensure that the segmentation is retained.
        - Get the loss against all rotated ground-truth maps.
        - apply the argmin over all losses against rotated maps.
- [ ] box fitting (Umeyama optimization)
    - The umeyama implementation from the original nocs work is copied into utils/aligning.py. 
    All that remains is using those to derive the box and pose metrics.s


## References

- **Pytorch example code:** pytorch has an example of DCGAN coded and explained 
[here](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

- **Diagnosing GAN training:** [This article](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/) 
goes over different failure modes and discusses the reason they occure and how to mitigate them.
