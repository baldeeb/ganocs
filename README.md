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
- [ ] box fitting (Umeyama optimization)
