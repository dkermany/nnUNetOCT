from typing import Tuple

import torch
import random

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform

class CustomNoiseTransform(ImageOnlyTransform):
    def __init__(self, sigma_range, block_range, artifact_label=11):
        super().__init__()
        self.sigma_range = sigma_range
        self.block_range = block_range 
        self.artifact_label = artifact_label

    def get_parameters(self, **data_dict) -> dict:
        segmentation = data_dict["segmentation"]
        artifact_mask = (segmentation == self.artifact_label) | (segmentation <= -1)
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        block_size = random.uniform(self.block_range[0], self.block_range[1])
        return {
            "artifact_mask": artifact_mask,
            "sigma": sigma,
            "block_size": round(block_size),
        }

    def _apply_to_image(self, image: torch.Tensor, **params) -> torch.Tensor:
        block_size = params["block_size"]
        sigma = params["sigma"]
        mask = params["artifact_mask"]

        # print(f"block_size: {block_size}")
        # print(f"input image shape: {image.shape}")
        # print(f"mask shape: {mask.shape}")

        noisy_image = image.clone()

        if block_size > 1:
            noise_shape = tuple((dim + block_size - 1) // block_size for dim in image.shape)
            # print(f"Block noise shape: {noise_shape}")
            noise = torch.normal(0, sigma, size=noise_shape, dtype=image.dtype, device=image.device)
            # print(f"initial noise shape: {noise.shape}")

            if len(image.shape) == 3:
                expanded_noise = noise.repeat_interleave(block_size, dim=2)
                expanded_noise = expanded_noise.repeat_interleave(block_size, dim=1)
                expanded_noise = expanded_noise[:image.shape[0], :image.shape[1], :image.shape[2]]
            elif len(image.shape) == 2:
                expanded_noise = noise.repeat_interleave(block_size, dim=1)
                expanded_noise = expanded_noise.repeat_interleave(block_size, dim=0)
                expanded_noise = expanded_noise[:image.shape[0], :image.shape[1]]
            noise = expanded_noise
            # print(f"Expanded noise shape: {noise.shape} (expected: {image.shape})")
        else:
            noise = torch.normal(0, sigma, size=image.shape, dtype=image.dtype, device=image.device)

        
        noise = noise[:image.shape[0], :image.shape[1]] if len(image.shape) == 2 else noise[:image.shape[0], :image.shape[1], :image.shape[2]]
        # print(f"Final noise shape: {noise.shape} (expected: {image.shape})")

        noisy_image[~mask] += noise[~mask]
        return torch.clamp(noisy_image, torch.min(image), torch.max(image))

