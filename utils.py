# -*- codeing = utf-8 -*-
# @Time : 2024/3/15 10:39
# @Author : 李昌杏
# @File : utils.py
# @Software : PyCharm
import torch
from torch import nn
import numpy as np
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



def mean_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = torch.logical_and(pred_inds, target_inds).sum().float()
        union = torch.logical_or(pred_inds, target_inds).sum().float()

        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou)

    mean_iou = torch.mean(torch.tensor(ious))
    return mean_iou.item()

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, early_schedule_epochs=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if early_schedule_epochs == 0:
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))
    else:
        iters = np.arange(early_schedule_epochs*niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        remainder = np.array([final_value]*((epochs - early_schedule_epochs) * niter_per_ep))
        schedule = np.concatenate((warmup_schedule, schedule, remainder))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
