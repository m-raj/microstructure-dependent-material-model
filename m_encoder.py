import torch.nn as nn
import torch.nn.functional as F
import torch


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Pad(nn.Module):
    def __init__(self, padding):
        super(Pad, self).__init__()
        self.padding = padding

    def forward(self, x):
        return nn.functional.pad(x, self.padding)


class JointAutoEncoder(nn.Module):
    def __init__(self, width, channels, latent_dim, kernel_size=3):
        super(JointAutoEncoder, self).__init__()
        self.encoder = nn.ModuleList()

        self.encoder.append(Pad((5, 6)))  # Pad input to handle odd lengths
        for i in range(len(channels) - 1):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels[i],
                        channels[i + 1],
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                )
            )

        encoder_bottleneck = nn.Linear(
            channels[-1] * (width // (2 ** (len(channels) - 1))), latent_dim
        )
        self.encoder.append(View((-1,)))
        self.encoder.append(encoder_bottleneck)

        decoder_bottleneck = nn.Linear(
            latent_dim, channels[-1] * (width // (2 ** (len(channels) - 1)))
        )

        self.decoder = nn.ModuleList()
        self.decoder.append(decoder_bottleneck)
        self.decoder.append(View((channels[-1], width // (2 ** (len(channels) - 1)))))
        for i in range(len(channels) - 1, 1, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(
                        channels[i],
                        channels[i - 1],
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(channels[i - 1]),
                    nn.ReLU(),
                )
            )

        self.decoder.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(
                    channels[1], channels[0], kernel_size, padding=kernel_size // 2
                ),
            )
        )

        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        x_recon = self.decoder(self.encoder(x))
        return x_recon[:, :, 6:-5]

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
