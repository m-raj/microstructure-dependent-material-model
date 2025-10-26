import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction:
    def __init__(self):
        super(LossFunction, self).__init__()

    def L2_norm(self, x):
        """L2 Norm"""
        return torch.square(torch.norm(x, dim=-1)).mean()

    def L2_error(self, predictions, targets):
        """L2 Error in time"""
        return self.L2_norm(predictions - targets)

    def relative_error(self, predictions, targets):
        """Relative Error"""
        return torch.sqrt(self.L2_error(predictions, targets) / self.L2_norm(targets))
