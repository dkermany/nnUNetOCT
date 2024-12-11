from typing import Tuple

import torch
import torch.nn.functional as F
import random

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform 

class CustomReshapeTransform(BasicTransform):
    def __init__(self, scale: RandomScalar, synchronize_channels: bool,
                 synchronize_axes: bool, allowed_channels: Tuple[int, ...]=None,
                 p_per_channel: float=1):
        super().__init__()
        self.scale = scale
        self.synchronize_channels = synchronize_channels
        self.synchronize_axes = synchronize_axes
        self.allowed_channels = allowed_channels
        self.p_per_channel = p_per_channel

    def reshape(self, image, target_height, target_width, mode="bilinear"):
        if len(image.shape) == 2: #2D Case
            reshaped_image = F.interpolate(
                image.unsqueeze(0).unsqueeze(0), # add batch and channel dims
                size=(target_height, target_width),
                mode=mode,
                align_corners=False,
            )
            return reshaped_image.squeeze(0).squeeze(0)
        elif len(image.shape) == 3: #3D Case
            slices = image.shape[0]
            reshaped_slices = []
            for i in range(slices):
                reshaped_slice = F.interpolate(
                    image[i].unsqueeze(0).unsqueeze(0), # add batch and channel dims
                    size=(target_height, target_width),
                    mode=mode,
                    align_corners=False,
                )
                reshaped_slices.append(reshaped_slice.squeeze(0).squeeze(0))

            # Stack reshaped slices back into tensor
            return torch.stack(reshaped_slices)
        else:
            raise ValueError("Image must be a 2D or 3D tensor")

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        if self.allowed_channels is None:
            apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        else:
            apply_to_channel = [i for i in self.allowed_channels if torch.rand(1) < self.p_per_channel]
        if self.synchronize_channels:
            if self.synchronize_axes:
                scales = torch.Tensor([[sample_scalar(self.scale, image=data_dict['image'], channel=None, dim=None)] * (len(shape) - 1)] * len(apply_to_channel))
            else:
                scales = torch.Tensor([[sample_scalar(self.scale, image=data_dict['image'], channel=None, dim=d) for d in range(len(shape) - 1)]] * len(apply_to_channel))
        else:
            if self.synchronize_axes:
                scales = torch.Tensor([[sample_scalar(self.scale, image=data_dict['image'], channel=c, dim=None)]  * (len(shape) - 1) for c in apply_to_channel])
            else:
                scales = torch.Tensor([[sample_scalar(self.scale, image=data_dict['image'], channel=c, dim=d) for d in range(len(shape) - 1)] for c in apply_to_channel])
        return {
            "apply_to_channel": apply_to_channel,
            "scales": scales,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        adjusted_slices = []
        for c, s in zip(params["apply_to_channel"], params["scales"]):
            adjusted_slices.append(self.reshape(img[c],
                                                img[c].shape[0],
                                                round(img[c].shape[1] * s[0].item())))
        
        adjusted_img = torch.stack(adjusted_slices)
        return adjusted_img

    def _apply_to_segmentation(self, seg: torch.Tensor, **params) -> torch.Tensor:
        adjusted_slices = []
        for c, s in zip(params["apply_to_channel"], params["scales"]):
            adjusted_slices.append(self.reshape(seg[c].to(torch.float32),
                                                seg[c].shape[0],
                                                round(seg[c].shape[1] * s[0].item())).to(torch.int16))
        
        adjusted_seg = torch.stack(adjusted_slices)
        return adjusted_seg
