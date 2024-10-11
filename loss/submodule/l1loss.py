from abc import ABC

import torch.nn as nn

class L1Loss(nn.Module, ABC):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = None

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss



