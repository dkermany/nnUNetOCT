from typing import Tuple

import torch
import random

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform

class CustomWindowLevelTransform(ImageOnlyTransform):
    def __init__(self,
                 window: RandomScalar,
                 synchronize_channels: bool,
                 synchronize_axes: bool,
                 ignore_axes: Tuple[int, ...],
                 allowed_channels: Tuple[int, ...] = None,
                 p_per_channel: float = 1):
        super().__init__()
        self.window = window
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
        if self.synchronize_channels:
            if self.synchronize_axes:
                windows = torch.Tensor([[sample_scalar(self.window, image=data_dict['image'], channel=None, dim=None)] * (len(shape) - 1)] * len(apply_to_channel))
            else:
                windows = torch.Tensor([[sample_scalar(self.window, image=data_dict['image'], channel=None, dim=d) for d in range(len(shape) - 1)]] * len(apply_to_channel))
        else:
            if self.synchronize_axes:
                windows = torch.Tensor([[sample_scalar(self.window, image=data_dict['image'], channel=c, dim=None)]  * (len(shape) - 1) for c in apply_to_channel])
            else:
                windows = torch.Tensor([[sample_scalar(self.window, image=data_dict['image'], channel=c, dim=d) for d in range(len(shape) - 1)] for c in apply_to_channel])

        return {
            'apply_to_channel': apply_to_channel,
            'windows': windows
        }

    def get_window_level_adjustment(self, image: torch.Tensor, window: int, level: int):
        """
        Apply random window and level adjustments to a 2D or 3D grayscale
        image
        """
        if len(image.shape) not in [2, 3]:
            raise ValueError("Input image must be 2D or 3D tensor")

        min_intensity = level - (window / 2)
        max_intensity = level + (window / 2)

        if len(image.shape) == 2: #2D Case
            adjusted_image = torch.clamp(image, min=min_intensity, max=max_intensity)
        else: #3D case
            # Process each slice independently
            adjusted_image = torch.zeros_like(image, dtype=torch.uint8)
            for i in range(image.shape[0]):
                slice_image = image[i]
                adjusted_slice = torch.clamp(slice_image, min=min_intensity, max=max_intensity)
                adjusted_image[i] = adjusted_slice

        return adjusted_image

    def get_level_value(self, window: int):
        if window == 10:
            level = random.randint(-4, 5)
        elif window == 100:
            level = random.randint(-49, 50)
        else:
            lower_bound = -4 - (window - 10) * (45 / 90)
            upper_bound = 5 + (window - 10) * (45 / 90)
            level = random.randint(round(lower_bound), round(upper_bound))

        return level

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        orig_shape = img.shape[1:]
        # we cannot batch this because the downsampled shaps will be different for each channel
        for c, w in zip(params['apply_to_channel'], params['windows']):
            w = round(w[0].item())
            l = self.get_level_value(w)
            img[c] = self.get_window_level_adjustment(img[c], w, l)

        return img
