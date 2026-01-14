import torch
import torch.nn as nn
import torch.nn.functional as F
from convex_network import *
from fnm import *

# Decompsition of W and D into three parts


class TrueFeatures(nn.Module):
    def __init__(self):
        super(TrueFeatures, self).__init__()

    def forward(self, feature1, feature2):

        nu_prime = torch.mean(feature2.pow(-1), dim=1, keepdim=True).pow(-1)
        E_prime = (
            torch.mean(feature1 / feature2.pow(2), dim=1, keepdim=True) * nu_prime**2
        )

        # nu_prime = torch.mean(feature2, dim=1, keepdim=True).pow(-1)
        # E_prime = torch.mean(feature1, dim=1, keepdim=True) * nu_prime**2
        output = torch.cat((E_prime, nu_prime), dim=-1)
        return output


class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return torch.square(F.relu(x))


class EnergyFunction(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(EnergyFunction, self).__init__()

        # self.E = nn.Sequential(
        #     nn.Linear(1002, hidden_dims[0]),
        #     nn.Softplus(),
        #     nn.Linear(hidden_dims[0], input_dim[0]),
        # )

        self.fnm1 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        )
        self.fnm2 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        )

        # self.B = nn.Sequential(
        #     nn.Linear(4, 50),
        #     CustomActivation(),
        #     nn.Linear(50, 1),
        # )

        self.nn = nn.Sequential(
            nn.Linear(4, 100), CustomActivation(), nn.Linear(100, 1)
        )

    def forward(self, u, v, m_features):

        features = torch.cat(
            (u, v, self.fnm1(m_features), self.fnm2(m_features)), dim=-1
        )

        energy = self.nn(features)

        return energy.squeeze(-1)

    def compute_derivative(self, u, v, m_features):
        u.requires_grad_(True)
        energy = self(u, v, m_features)
        grad_u, grad_v = torch.autograd.grad(
            energy.sum(),
            (u, v),
            create_graph=True,
            retain_graph=True,
        )
        return grad_u, grad_v


class InverseDissipationPotential(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(InverseDissipationPotential, self).__init__()
        # self.nu = nn.Sequential(
        #     nn.Linear(501, hidden_dims[0]),
        #     nn.Softplus(),
        #     nn.Linear(hidden_dims[0], input_dim[0]),
        # )

        self.fnm1 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=1, d_out=1, n_layers=3
        )
        self.fnm2 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        )

        self.nn = nn.Sequential(nn.Linear(1, 50), CustomActivation(), nn.Linear(50, 1))

        self.dissipation = ConvexNetwork(1, 50)

    def forward(self, p, q, m_features):
        p.requires_grad_(True)
        _, nu = torch.split(m_features, [1, 1], dim=1)

        potential = -1 / 2 * self.fnm1(nu) * self.nn(p) + 1 / 2 * torch.sum(
            self.fnm2(m_features) * self.dissipation(q), dim=-1, keepdim=True
        )
        return potential.squeeze(-1)

    def compute_derivative(self, p, q, m_features):
        potential = self(p, q, m_features)
        grad_p, grad_q = torch.autograd.grad(
            potential.sum(), (p, q), create_graph=True, retain_graph=True
        )
        return grad_p, grad_q


class ViscoelasticMaterialModel(nn.Module):
    def __init__(
        self,
        energy_input_dim=None,
        energy_hidden_dim=None,
        dissipation_input_dim=None,
        dissipation_hidden_dim=None,
        E_encoder=None,
        nu_encoder=None,
        dt=None,
    ):
        super(ViscoelasticMaterialModel, self).__init__()
        self.niv = energy_input_dim[1]
        self.energy_function = EnergyFunction(energy_input_dim, energy_hidden_dim)
        self.dissipation_potential = InverseDissipationPotential(
            dissipation_input_dim, dissipation_hidden_dim
        )
        self.dt = dt  # Time step size

        # Microstructure encoder
        self.tf = TrueFeatures()
        self.E_encoder = E_encoder
        self.nu_encoder = nu_encoder

    def microstructure_encoder(self, E, nu):
        # feat = self.tf(E, nu)
        x = torch.stack((E, nu), dim=1)
        return x

    def forward(self, e, e_dot, E=None, nu=None):
        m_features = self.microstructure_encoder(E, nu)
        stress = []
        xi = [
            torch.zeros(
                e.shape[0], self.niv, requires_grad=True, dtype=e.dtype, device=e.device
            )
        ]
        for i in range(0, e.shape[1]):
            s_eq, d = self.compute_energy_derivative(e[:, i], xi[i], m_features)
            s_neq, kinetics = self.compute_dissipation_derivative(
                e_dot[:, i], -d, m_features
            )
            stress.append(s_eq - s_neq)
            if i < e.shape[1] - 1:
                xi.append(xi[-1] + self.dt * kinetics)
        stress = torch.stack(stress, dim=1)
        xi = torch.stack(xi, dim=1)
        return stress, xi

    def compute_energy_derivative(self, u, v, m_features):
        return self.energy_function.compute_derivative(u, v, m_features)

    def compute_dissipation_derivative(self, p, q, m_features):
        return self.dissipation_potential.compute_derivative(p, q, m_features)


# def train_step(model, optimizer, e, e_dot, E, nu, s_true):
#     model.train()
#     optimizer.zero_grad()
#     s_pred, _ = model(e, e_dot, E, nu)
#     loss = F.mse_loss(s_pred, s_true)
#     loss.backward()
#     optimizer.step()
#     return loss.item()


# def prediction_step(model, e, e_dot, E, nu):
#     model.eval()
#     s_pred, xi = model(e, e_dot, E, nu)
#     return s_pred, xi
