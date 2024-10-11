from abc import ABC

import torch.nn as nn


class L2Loss(nn.Module, ABC):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = None

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]

        self.loss = (diff**2).mean()

        return self.loss

