# Pytorch NOCS & Domain Adaptation

## Contributions

- Pytorch implementation of [NOCS](https://arxiv.org/pdf/1901.02970.pdf) ([code](https://github.com/hughw19/NOCS_CVPR2019/tree/master)) based off of the [torchvision maskrcnn](https://pytorch.org/vision/main/models/mask_rcnn.html) implementation.
  - Includes [WandB](https://wandb.ai/) logging.
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
- [ ] Speed up NOCS to depth optimization using GPU.
- [ ] Adjusting non-max-supression on inference time.
- [ ] 3D bbox IoU eval metric.
- [ ] Moving habitate datagen folder to datasets

## References

- GAN implementation:
  - **Pytorch example code:** pytorch has an example of DCGAN coded and explained [here](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
  - **Diagnosing GAN training:** [This article](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/) goes over different failure modes and discusses the reason they occure and how to mitigate them.

- Alignment implementation
  - parts were taken from the original nocs code 
  - [BYOC](https://github.com/mbanani/byoc) implementation was used to provide better optimized implementation.