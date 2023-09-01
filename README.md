# Pytorch NOCS & Domain Adaptation

## Contributions: 
- Pytorch implementation of [NOCS]() ([code]()) based off of the torchvision maskrcnn implementation.
- WandB logging and Evaluation.
- Experiments performing domain adaptation:
  - Tools for integrating auxiliary losses
  - Generative Adversarial Network as NOCS head.
- Dataloaders for different sources:
  - Original dataset from NOCS.
  - Habitat based data generator/loader (Preliminary/Rushed implementation).
  - Rosbag loaders to use ROSbag data for testing models on robots.
- Inference wrapper around the whole model.

## Pending implementations

- [ ] NOCS symmetry loss.
- [ ] Speed up NOCS to depth optimization.

This is a re-implementation of the NOCS paper [link](TODO) over pytorch's [maskrcnn](TODO) implementation.

The model is verified on a dataset collected using [shapenet](TODO) and the [Habitat](TODO) simulator

## References

- GAN implementation:
  - **Pytorch example code:** pytorch has an example of DCGAN coded and explained [here](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
  - **Diagnosing GAN training:** [This article](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/) goes over different failure modes and discusses the reason they occure and how to mitigate them.
