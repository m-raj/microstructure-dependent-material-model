import torch.nn as nn
import torch.nn.functional as F
import torch


class BilinearInput(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(BilinearInput, self).__init__()
        self.parameter = nn.Parameter(torch.randn(input_dim1, input_dim2, output_dim))

    def forward(self, x):
        """x: (batch_size, input_dim1, input_dim2)"""
        output = torch.einsum("bij,ijk->bk", x, self.parameter)
        return output


class BilinearOutput(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2):
        super(BilinearOutput, self).__init__()
        self.parameter = nn.Parameter(torch.randn(input_dim, output_dim1, output_dim2))

    def forward(self, x):
        """x: (batch_size, input_dim)"""
        output = torch.einsum("bi,ijk->bjk", x, self.parameter)
        return output


class JointAutoEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dim):
        super(JointAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            BilinearInput(input_dims[0], input_dims[1], hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            BilinearOutput(hidden_dims, input_dims[0], input_dims[1]),
            nn.Sigmoid(),
        )

    def forward(self, E, nu):
        x = torch.stack((E, nu), dim=-1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        E_recon, nu_recon = torch.split(reconstructed, [1, 1], dim=-1)
        return E_recon.squeeze(-1), nu_recon.squeeze(-1)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True


class PCAEncoder(nn.Module):
    def __init__(
        self, input_dim, latent_dim, components=None, mean=None, data_files=None
    ):
        super(PCAEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.components = components
        self.mean = mean
        self.data_files = data_files
        self.initialize_weights(latent_dim)

    def initialize_weights(self, latent_dim):
        components = self.components[:latent_dim]
        self.encoder.weight.data = components
        self.encoder.bias.data = -self.mean @ components.t()

    def decoder(self, z):
        return (z - self.encoder.bias.data) @ self.encoder.weight.data

    def forward(self, x):
        latent = self.encoder(x)
        return latent

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    reconstructed = model(data)
    loss = F.mse_loss(reconstructed, data)
    loss.backward()
    optimizer.step()
    return loss.item()
