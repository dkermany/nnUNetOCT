from typing import Tuple

import torch
import random

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform

class CustomArtifactTransform(ImageOnlyTransform):
    def __init__(self, version: Tuple[str, ...], artifact_label=11):
        super().__init__()
        self.version = version 
        self.artifact_label = artifact_label

    def get_parameters(self, **data_dict) -> dict:
        segmentation = data_dict["segmentation"]
        artifact_mask = (segmentation == self.artifact_label) | (segmentation <= -1)
        selected_version = random.choice(self.version)
        return {
            "artifact_mask": artifact_mask,
            "selected_version": selected_version,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        version = params["selected_version"]
        artifact_mask = params["artifact_mask"]

        min_value = img.min()
        max_value = img.max()
        if version == "white":
            img[artifact_mask] = max_value
        elif version == "black":
            img[artifact_mask] = min_value
        else:
            # random gray value between white and black
            img[artifact_mask] = torch.rand(1).item() * (max_value - min_value) + min_value
    
        return img

