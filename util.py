import torch, pickle
import torch.nn.functional as F
from torch.utils.data import Dataset


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


class ViscoelasticDataset(Dataset):
    def __init__(self, data_path, N, step, device="cpu"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.e = torch.tensor(data["strain"][:N, ::step], dtype=torch.float32)
        self.e_dot = torch.tensor(data["strain_rate"][:N, ::step], dtype=torch.float32)
        self.s = torch.tensor(data["stress"][:N, ::step], dtype=torch.float32)

        self.E = torch.tensor(data["E"][:N], dtype=torch.float32)
        self.nu = torch.tensor(data["nu"][:N], dtype=torch.float32)

        self.device = device

    def __len__(self):
        return len(self.e)

    def __getitem__(self, idx):
        x = (
            self.e[idx].to(self.device),
            self.e_dot[idx].to(self.device),
            self.E[idx].to(self.device),
            self.nu[idx].to(self.device),
        )
        y = self.s[idx].to(self.device)
        return x, y
