# TriNerSeg
Segmentation of Trigeminal Nerve and Cerebrovasculature in MR-Angiography Images

- The model is defined in ```model/coarse2fine.py```
- ```train3dv2.py``` to train network
- ```predict3d.py``` to predict segmentations

## Requirements
- [TorchIO](https://torchio.readthedocs.io/)
- [SimpleITK](https://pypi.org/project/SimpleITK/)
- [PyTorch >= 1.5.0](https://pytorch.org/)
