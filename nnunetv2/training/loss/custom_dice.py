from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


class CustomSoftDiceLoss(nn.Module):
    def __init__(self,
                 apply_nonlin: Callable = None,
                 batch_dice: bool = False,
                 do_bg: bool = True,
                 smooth: float = 1.,
                 ddp: bool = True,
                 clip_tp: float = None,
                 weight: torch.Tensor = None):
        """
        """
        super(CustomSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp
        self.weight = weight

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc_per_class = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        # Handle background class if do_bg=False
        if not self.do_bg:
            if self.batch_dice:
                # batch_dice=True -> dc_per_class is [C], remove bg class at index 0
                dc_per_class = dc_per_class[1:]
                weight_slice = self.weight[1:] if self.weight is not None else None
            else:
                # batch_dice=False -> dc_per_class is [N, C], remove bg class at index 0
                dc_per_class = dc_per_class[:, 1:]
                weight_slice = self.weight[1:] if self.weight is not None else None
        else:
            weight_slice = self.weight

        # Apply class weights if provided
        if weight_slice is not None:
            weight_slice = weight_slice.to(dc_per_class.device)
            if self.batch_dice:
                # dc_per_class is [C] (or [C-1] if do_bg=False)
                # Weighted mean across classes
                dc = (dc_per_class * weight_slice).sum() / weight_slice.sum()
            else:
                # dc_per_class is [N, C] (or [N, C-1] if do_bg=False)
                # Weighted mean over classes for each sample, then mean over batch
                dc_sample = (dc_per_class * weight_slice.unsqueeze(0)).sum(dim=1) / weight_slice.sum()
                dc = dc_sample.mean()
        else:
            # No class weights: original behavior
            dc = dc_per_class.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn

