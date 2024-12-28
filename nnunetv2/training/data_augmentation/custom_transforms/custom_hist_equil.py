from typing import Tuple

import torch
import random
import cv2
import kornia

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform

class CustomHistEquilTransform(ImageOnlyTransform):
    def __init__(self,
                 synchronize_channels: bool,
                 synchronize_axes: bool,
                 ignore_axes: Tuple[int, ...],
                 allowed_channels: Tuple[int, ...] = None,
                 p_per_channel: float = 1):
        super().__init__()
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes
        self.ignore_axes = ignore_axes
        self.allowed_channels = allowed_channels
        self.p_per_channel = p_per_channel

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        if self.allowed_channels is None:
            apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        else:
            apply_to_channel = [i for i in self.allowed_channels if torch.rand(1) < self.p_per_channel]

        return {
            'apply_to_channel': apply_to_channel,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        orig_shape = img.shape[1:]
        for c in zip(params['apply_to_channel']):
            img[c] = kornia.enhance.equalize(img[c])

        return img
