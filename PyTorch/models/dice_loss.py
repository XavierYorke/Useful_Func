#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Time    :   2022/3/29
# Author  :   XavierYorke
# Contact :   mzlxavier1230@gmail.com

import torch
import torch.nn as nn


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)  # 获得图像的维数
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)  # 将维数的数据转换到第一位
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"

        output = flatten(output)
        target = flatten(target)

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)

        dice = ((2. * intersect) + self.epsilon) / (denominator + self.epsilon)
        dice = torch.mean(dice)
        return 1 - dice


class DiceLoss2(nn.Module):
    def __init__(self):
        super(DiceLoss2, self).__init__()
        self.epsilon = 1e-5

    def forward(self, output, h3, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"

        output = flatten(output)
        target = flatten(target)

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)

        h3 = flatten(h3)
        intersect2 = (h3 * target).sum(-1)
        denominator2 = (h3 + target).sum(-1)

        dice2 = ((2. * intersect2) + self.epsilon) / (denominator2 + self.epsilon)
        dice2 = torch.mean(dice2)

        dice2 = 1 - dice2

        dice = ((2. * intersect) + self.epsilon) / (denominator + self.epsilon)
        dice = torch.mean(dice)
        return 1 - dice + 0.5 * dice2


class DiceLoss1(nn.Module):
    def __init__(self):
        super(DiceLoss1, self).__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"

        output = flatten(output)
        target = flatten(target)

        out = output.sum(-1)

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)

        false_re = out - intersect
        ratio = (out - intersect + self.epsilon) / (target.sum(-1) + self.epsilon)

        dice = ((2. * intersect) + self.epsilon) / (denominator + self.epsilon + (1 - torch.exp(-ratio)) * false_re)
        dice = torch.mean(dice)
        return 1 - dice


def dice_cal(output, target):
    epsilon = 1e-5
    assert output.size() == target.size(), "'input' and 'target' must have the same shape"

    output = flatten(output)
    target = flatten(target)

    intersect = (output * target).sum(-1)
    denominator = (output + target).sum(-1)
    dice = ((2. * intersect) + epsilon) / (denominator + epsilon)
    dice = torch.mean(dice)
    return dice

