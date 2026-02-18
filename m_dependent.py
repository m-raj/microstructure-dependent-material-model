import torch
import torch.nn as nn
import torch.nn.functional as F
from convex_network import *
from fnm import *

# Decompsition of W and D into three parts


class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return torch.square(F.relu(x))


class EnergyFunction(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=50):
        super(EnergyFunction, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            CustomActivation(),
            nn.Linear(hidden_dim, 1),
        )

        # self.picnn = PartiallyInputConvex(
        #    y_dim=1, x_dim=3, z_dim=hidden_dim, u_dim=hidden_dim, bias1=True, bias2=True
        # )
        # self.icnn = ConvexNetwork(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, u, v, m_features):
        energy = self.nn(torch.cat((u, v, *m_features), dim=-1))
        # energy = self.picnn(u, torch.cat((v, *m_features), dim=-1))
        # energy = self.icnn(torch.cat(u, v, *m_features, dim=-1))

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
    def __init__(self):
        super(InverseDissipationPotential, self).__init__()
        self.picnn1 = PartiallyInputConvex(
            y_dim=1, x_dim=1, z_dim=50, u_dim=50, bias1=True, bias2=True
        )
        self.picnn2 = PartiallyInputConvex(
            y_dim=1, x_dim=1, z_dim=50, u_dim=50, bias1=True, bias2=True
        )

    def forward(self, p, q, m_features):
        p.requires_grad_(True)
        features1, features2 = m_features
        potential = -self.picnn1(p, features1) + self.picnn2(q, features2)
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
        dt=None,
    ):
        super(ViscoelasticMaterialModel, self).__init__()
        self.niv = energy_input_dim[1]
        self.energy_function = EnergyFunction(energy_input_dim, energy_hidden_dim)
        self.dissipation_potential = InverseDissipationPotential(
            dissipation_input_dim, dissipation_hidden_dim
        )
        self.dt = dt  # Time step size

        self.fnm1 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        )
        self.fnm2 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        )

        self.fnm3 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=1, d_out=1, n_layers=3
        )
        self.fnm4 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        )

    def microstructure_encoder(self, E, nu):
        microstructure = torch.stack((E, nu), dim=1)
        features1 = self.fnm1(microstructure)
        features2 = self.fnm2(microstructure)
        features3 = self.fnm3(nu)
        features4 = self.fnm4(microstructure)
        return (features1, features3), (features2, features4)

    def forward(self, e, e_dot, E=None, nu=None):

        stress = []
        xi = [
            torch.zeros(
                e.shape[0], self.niv, requires_grad=True, dtype=e.dtype, device=e.device
            )
        ]
        m_features1, m_features2 = self.microstructure_encoder(E, nu)

        for i in range(0, e.shape[1]):
            s_eq, d = self.compute_energy_derivative(e[:, i], xi[i], m_features1)
            s_neq, kinetics = self.compute_dissipation_derivative(
                e_dot[:, i], -d, m_features2
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
