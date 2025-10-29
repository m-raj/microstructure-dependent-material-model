import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(torch.nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def L2NormSquared(self, x):
        "x : [B, T, D]"
        return torch.mean(torch.sum(torch.square(x), dim=-1), dim=1)

    def L2Norm(self, x):
        "x : [B, T, D]"
        return torch.sqrt(self.L2NormSquared(x))

    def L2RelativeError(self, x, y, reduction="mean"):
        error = x - y
        rel_error = self.L2Norm(error) / (self.L2Norm(y) + 1.0e-8)
        if reduction == "mean":
            return torch.mean(rel_error)
        elif not (reduction):
            return rel_error
        else:
            print("Not a valid reduction method: " + reduction)
