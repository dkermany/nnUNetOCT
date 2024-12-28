import torch
from typing import Tuple, Union, List
import numpy as np

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.training.data_augmentation.custom_transforms.custom_low_resolution import CustomLowResolutionTransform
from nnunetv2.training.data_augmentation.custom_transforms.custom_window_level import CustomWindowLevelTransform
from nnunetv2.training.data_augmentation.custom_transforms.custom_artifact_transform import CustomArtifactTransform
from nnunetv2.training.data_augmentation.custom_transforms.custom_spatial_transform import CustomSpatialTransform
from nnunetv2.training.data_augmentation.custom_transforms.custom_noise_transform import CustomNoiseTransform

from nnunetv2.training.loss.custom_compound_losses import Custom_DC_and_CE_loss
from nnunetv2.training.loss.custom_compound_losses import Custom_DC_and_BCE_loss

class CustomNNUNetTrainer4(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(
            plans, configuration, fold, dataset_json=dataset_json, unpack_dataset=unpack_dataset,
            device=device
        )

        print("Using CustomNNUNetTrainer!")
        #                                  bg    PED   HRF   FLU   HTD   RPE   RET  CHO  VIT  HYA  SHS  ART  ERM  SES
        self.class_weights = torch.tensor([1.00, 3.00, 4.00, 1.75, 2.13, 1.00, 0.5, 0.1, 0.3, 3.5, 1.0, 1.0, 1.0, 1.5],
                                          dtype=torch.float32).to(device) 

    def _build_loss(self):
        ## Adding custom class weights for OCT feature segmentation

        if self.label_manager.has_regions:
            print("Using DC_and_BCE_loss")
            loss = Custom_DC_and_BCE_loss({'weight': self.class_weights},
                                          {'batch_dice': self.configuration_manager.batch_dice,
                                           'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp,
                                           'weight': self.class_weights},
                                          use_ignore_label=self.label_manager.ignore_label is not None)
                                          #dice_class=MemoryEfficientSoftDiceLoss)
        else:
            print("Using DC_and_CE_loss")
            loss = Custom_DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                          'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp, 'weight': self.class_weights},
                                         {'weight': self.class_weights},
                                         weight_ce=1, weight_dice=1, ignore_label=self.label_manager.ignore_label)
                                         #dice_class=MemoryEfficientSoftDiceLoss)

        ## No custom edits beyond this point

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # transforms.append(
        #     CustomSpatialTransform(
        #         patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
        #         p_rotation=0.2,
        #         rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.85, 1.1), p_synchronize_scaling_across_axes=1,
        #         bg_style_seg_sampling=False  # , mode_seg='nearest'
        #     )
        # )
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.9, 1.1), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

       # transforms.append(RandomTransform(
       #     CustomNoiseTransform(
       #         sigma_range=(0.2, 0.7),
       #         block_range=(0, 7),
       #     ), apply_probability=0.2
       # ))
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))

        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.25
        ))
        # Added CustomLowResTransform
        transforms.append(RandomTransform(
            CustomLowResolutionTransform(
                scale=(0.1, 0.5),
                synchronize_channels=True,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=1
            ), apply_probability=0.33
        ))
        # Added CustomWindowLevelTransform
        # transforms.append(RandomTransform(
        #     CustomWindowLevelTransform(
        #         window=(10, 100),
        #         synchronize_channels=True,
        #         synchronize_axes=True,
        #         ignore_axes=ignore_axes,
        #         allowed_channels=None,
        #         p_per_channel=1
        #     ), apply_probability=0.33
        # ))
        # Added CustomArtifactTransform
        transforms.append(RandomTransform(
            CustomArtifactTransform(
                version=("black"),
                artifact_label=11,
            ), apply_probability=0.5
        ))
        # Added CustomReshapeTransform
        # but doesn't work because all image shapes need to be the same
        # will instead try reshaping all inference images to training size
        # instead
        # transforms.append(RandomTransform(
        #     CustomReshapeTransform(
        #         scale=(0.33, 1.0),
        #         synchronize_channels=True,
        #         synchronize_axes=True,
        #         allowed_channels=None,
        #         p_per_channel=1
        #     ), apply_probability=0.2
        # ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=True,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=True,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
