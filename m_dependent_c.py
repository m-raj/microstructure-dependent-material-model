import torch
import torch.nn as nn
import torch.nn.functional as F
from convex_network import *


class EnergyFunctionM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnergyFunctionM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, u, v, m_features):
        x = torch.cat((u, v, m_features), dim=-1)
        x = torch.square(F.relu(self.fc1(x)))
        energy = self.fc2(x)
        return energy.squeeze(-1)

    def compute_derivative(self, u, v, m_features):
        u.requires_grad_(True)
        energy = self(u, v, m_features)
        grad_u, grad_v = torch.autograd.grad(
            energy.sum(), (u, v), create_graph=True, retain_graph=True
        )
        return grad_u, grad_v


class InverseDissipationPotentialM(nn.Module):
    def __init__(self, y_dim, x_dim, hidden_dim):
        super(InverseDissipationPotentialM, self).__init__()

        self.picnn = PartiallyInputConvex(y_dim, x_dim, hidden_dim, hidden_dim)

    def forward(self, p, q, m_features):
        p.requires_grad_(True)
        m_features = self.microstructure_encoder(m_features)
        x = torch.cat((p, m_features), dim=-1)
        potential = self.picnn(q, x)
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
        energy_input_dim,
        energy_hidden_dim,
        dissipation_input_dim,
        dissipation_hidden_dim,
        E_encoder,
        nu_encoder,
        dt,
    ):
        super(ViscoelasticMaterialModel, self).__init__()
        self.energy_function = EnergyFunctionM(
            sum(energy_input_dim), energy_hidden_dim, E_encoder, nu_encoder
        )
        y_dim, x_dim = (
            dissipation_input_dim[1],
            dissipation_input_dim[0] + dissipation_input_dim[2],
        )
        self.dissipation_potential = InverseDissipationPotentialM(
            y_dim, x_dim, dissipation_hidden_dim, E_encoder, nu_encoder
        )
        self.dt = dt  # Time step size
        self.niv = energy_input_dim[1]

        # Microstructure encoder
        self.E_encoder = E_encoder
        self.nu_encoder = nu_encoder

    def microstructure_encoder(self, E, nu):
        x = torch.cat((self.E_encoder(E), self.nu_encoder(nu)), dim=-1)
        return x

    def forward(self, e, e_dot, E, nu):
        stress = []
        m_features = self.microstructure_encoder(E, nu)
        xi = [torch.zeros_like(torch.zeros(e.shape[0], self.niv), requires_grad=True)]
        s_eq, d = self.compute_energy_derivative(e[:, 0], xi[0], m_features)
        for i in range(1, e.shape[1]):
            s_neq, kinetics = self.compute_dissipation_derivative(
                e_dot[:, i - 1], -d, m_features
            )
            xi.append(xi[-1] + self.dt * kinetics)
            stress.append(s_eq - s_neq)
            s_eq, d = self.compute_energy_derivative(e[:, i], xi[-1], m_features)
        stress.append(s_eq - s_neq)
        stress = torch.stack(stress, dim=1)
        xi = torch.stack(xi, dim=1)
        return stress, xi

    def compute_energy_derivative(self, u, v, m_features):
        return self.energy_function.compute_derivative(u, v, m_features)

    def compute_dissipation_derivative(self, p, q, m_features):
        return self.dissipation_potential.compute_derivative(p, q, m_features)


def train_step(model, optimizer, e, e_dot, E, nu, s_true):
    model.train()
    optimizer.zero_grad()
    s_pred, _ = model(e, e_dot, E, nu)
    loss = F.mse_loss(s_pred, s_true)
    loss.backward()
    optimizer.step()
    return loss.item()


def prediction_step(model, e, e_dot, E, nu):
    model.eval()
    s_pred, xi = model(e, e_dot, E, nu)
    return s_pred, xi
