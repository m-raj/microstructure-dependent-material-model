import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.ReLU(),
#            nn.Linear(hidden_dims, hidden_dims),
#            nn.ReLU(),
            nn.Linear(hidden_dims, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),
#            nn.Linear(hidden_dims, hidden_dims),
#            nn.ReLU(),
            nn.Linear(hidden_dims, input_dim),
            nn.Sigmoid(),
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
