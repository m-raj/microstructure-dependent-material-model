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

    def L2RelativeError(self, pred, true, reduction="mean"):
        error = pred - true
        rel_error = self.L2Norm(error) / (self.L2Norm(true) + 1.0e-8)
        if reduction == "mean":
            return torch.mean(rel_error)
        if reduction == "sum":
            return torch.sum(rel_error)
        elif not (reduction):
            return rel_error
        else:
            print("Not a valid reduction method: " + reduction)


class ViscoelasticDataset(Dataset):
    def __init__(self, data_path, step, onebynu=False, device="cpu", encoder=False):

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.e = torch.tensor(data["strain"][:, ::step], dtype=torch.float32)
        self.e_dot = torch.tensor(data["strain_rate"][:, ::step], dtype=torch.float32)
        self.s = torch.tensor(data["stress"][:, ::step], dtype=torch.float32)

        self.E = torch.tensor(data["E"], dtype=torch.float32)
        self.nu = torch.tensor(data["nu"], dtype=torch.float32)

        self.device = device

        self.encoder = encoder

        self.E_stats = torch.load("data/mixture_random_field_process_E_stats.pt")
        self.nu_stats = torch.load("data/mixture_random_field_process_nu_stats.pt")
        self.onebynu = onebynu

    def __len__(self):
        return len(self.e)

    def transform(self, E, nu):
        E_norm = (E - self.E_stats["mean"]) / self.E_stats["std"]
        if not (self.onebynu):
            nu_norm = (nu - self.nu_stats["mean"]) / self.nu_stats["std"]
        else:
            nu_norm = (1.0 / nu - self.nu_stats["mean"]) / self.nu_stats["std"]
        return E_norm, nu_norm

    def __getitem__(self, idx):
        x = (
            self.e[idx].to(self.device),
            self.e_dot[idx].to(self.device),
            *self.transform(self.E[idx], self.nu[idx]),
        )
        y = self.s[idx].to(self.device)
        return x, y
