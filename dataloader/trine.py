"""
Dataloader for Trigeminal Nerve segmentation task
"""
from __future__ import print_function, division
import random
import numpy as np
import SimpleITK as sitk
import torchio

import os
import glob
import torch
from torch.utils.data import Dataset


def load_dataset(root_dir, train=True):
    if train:
        sub_dir = 'training'
    else:
        sub_dir = 'test'
    images_path = os.path.join(root_dir, sub_dir, 'images')
    labels_path = os.path.join(root_dir, sub_dir, 'labels')

    images = sorted(glob.glob(os.path.join(images_path, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(labels_path, "*.nii.gz")))

    return images, labels


class Data(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        self.images, self.groundtruth = load_dataset(self.root_dir, self.train)

        if self.train:
            self.transform = torchio.Compose([
                torchio.RescaleIntensity((0, 1)),
                torchio.RandomFlip(p=0.5),
                torchio.RandomNoise(p=0.25),
                torchio.RandomAffine(p=0.5),
            ])
        else:
            self.transform = torchio.Compose([
                torchio.RescaleIntensity((0, 1)),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.groundtruth[idx]
        subject = torchio.Subject(
            image=torchio.ScalarImage(image_path),
            label=torchio.LabelMap(label_path),
        )
        transformed = self.transform(subject)
        image = transformed['image'][torchio.DATA].numpy().astype(np.float32)
        label = transformed['label'][torchio.DATA].numpy().astype(np.int64)

        image = torch.from_numpy(np.ascontiguousarray(image))
        label = torch.from_numpy(np.ascontiguousarray(label))

        return image, label
