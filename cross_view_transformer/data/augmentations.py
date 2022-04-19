"""
https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/transforms.py
"""
import imgaug.augmenters as iaa
import torchvision
import numpy as np


class AugBase(torchvision.transforms.ToTensor):
    def __init__(self):
        super().__init__()

        self.augment = self.get_augment().augment_image

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        return self.augment(x)


class StrongAug(AugBase):
    def get_augment(self):
        return iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
        ])


class GeometricAug(AugBase):
    def get_augment(self):
        return iaa.Affine(rotate=(-2.5, 2.5),
                          translate_percent=(-0.05, 0.05),
                          scale=(0.95, 1.05),
                          mode='symmetric')
