# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import src.gridtransforms as gridsample_transforms
import torch
logger = getLogger()


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class MultiCropDatasetGrid(MultiCropDataset):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        grid_size=7
    ):
        super(MultiCropDatasetGrid, self).__init__(
            data_path,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            size_dataset,
            return_index,
        )
        ### modify for pixel-level begin ###
        self.grid_size = grid_size

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(0, len(size_crops)):
            randomresizedcrop = gridsample_transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend(
                [
                    gridsample_transforms.Compose(
                        [
                            randomresizedcrop,
                            gridsample_transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Compose(color_transform),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ]
                    )
                ]
                * nmb_crops[i]
            )
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))

        imgs = []
        for crop in multi_crops:
            imgs.append(crop[0])

        rectq = multi_crops[0][1]  # [left, top, width, height]
        rectk = multi_crops[1][1]

        gridq, gridk = get_grid(rectq, rectk, self.grid_size)

        return imgs, gridq, gridk


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def get_grid(rectq, rectk, size):
    grid = float(size - 1)
    overlap = [
        max(rectq[0], rectk[0]),
        max(rectq[1], rectk[1]),
        min(rectq[0] + rectq[2], rectk[0] + rectk[2]),
        min(rectq[1] + rectq[3], rectk[1] + rectk[3]),
    ]
    if overlap[0] < overlap[2] and overlap[1] < overlap[3]:
        q_overlap = torch.FloatTensor(
            [
                (overlap[0] - rectq[0]) / rectq[2],
                (overlap[1] - rectq[1]) / rectq[3],
                (overlap[2] - overlap[0]) / rectq[2],
                (overlap[3] - overlap[1]) / rectq[3],
            ]
        )
        k_overlap = torch.FloatTensor(
            [
                (overlap[0] - rectk[0]) / rectk[2],
                (overlap[1] - rectk[1]) / rectk[3],
                (overlap[2] - overlap[0]) / rectk[2],
                (overlap[3] - overlap[1]) / rectk[3],
            ]
        )

        q_grid = torch.zeros(size=(size, size, 2), dtype=torch.float32)
        k_grid = torch.zeros(size=(size, size, 2), dtype=torch.float32)
        q_grid[:, :, 0] = torch.FloatTensor(
            [q_overlap[0] + i * q_overlap[2] / grid for i in range(size)]
        ).view(1, size)
        q_grid[:, :, 1] = torch.FloatTensor(
            [q_overlap[1] + i * q_overlap[3] / grid for i in range(size)]
        ).view(size, 1)
        k_grid[:, :, 0] = torch.FloatTensor(
            [k_overlap[0] + i * k_overlap[2] / grid for i in range(size)]
        ).view(1, size)
        k_grid[:, :, 1] = torch.FloatTensor(
            [k_overlap[1] + i * k_overlap[3] / grid for i in range(size)]
        ).view(size, 1)

        # flip
        if rectq[4] > 0:
            q_grid[:, :, 0] = 1 - q_grid[:, :, 0]

        if rectk[4] > 0:
            k_grid[:, :, 0] = 1 - k_grid[:, :, 0]

        k_grid = 2 * k_grid - 1
        q_grid = 2 * q_grid - 1

    else:
        # fill zero
        q_grid = torch.full(fill_value=-2, size=(size, size, 2), dtype=torch.float32)
        k_grid = torch.full(fill_value=-2, size=(size, size, 2), dtype=torch.float32)

    return q_grid, k_grid
