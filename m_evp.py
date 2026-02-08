import torch
import torch.nn as nn
import torch.nn.functional as F
from convex_network import *
from fnm import *
import tqdm

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

    def forward(self, u, v, m_features):
        energy = 0.5 * m_features * (u - v) ** 2
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
    def __init__(self, out_dim, z_dim, u_dim):
        super(InverseDissipationPotential, self).__init__()
        self.picnn1 = PartiallyInputConvex(y_dim=1, x_dim=out_dim, z_dim=z_dim, u_dim=u_dim)
        # self.picnn2 = PartiallyInputConvex(y_dim=1, x_dim=1, z_dim=50, u_dim=50)
        # self.dense_network = nn.Sequential(
        #     nn.Linear(10, 1000), CustomActivation(), nn.Linear(1000, 1)
        # )

    def forward(self, q, m_features):

        features = torch.cat((q, m_features), dim=1)
        # potential = self.dense_network(features)

        potential = self.picnn1(q, m_features)

        # Y, n, edot_0 = torch.split(m_features, 1, dim=1)
        # Y, n, edot_0 = Y.squeeze(), n.squeeze(), edot_0.squeeze()
        # potential = torch.mean(
        #     torch.pow(torch.abs(q), n + 1) / (n + 1) * edot_0 * torch.pow(Y, -n),
        #     dim=1,
        #     keepdim=True,
        # )
        return potential.squeeze(-1)

    def compute_derivative(self, q, m_features):

        potential = self(q, m_features)
        grad_q = torch.autograd.grad(
            potential.sum(), q, create_graph=True, retain_graph=True
        )[0]
        return grad_q


class ViscoplasticMaterialModel(nn.Module):
    def __init__(
        self,
        energy_input_dim=None,
        energy_hidden_dim=None,
        dissipation_input_dim=None,
        dissipation_hidden_dim=None,
        dt=None,
        out_dim=None,
        modes=None,
        z_dim=None,
        u_dim=None
    ):
        super(ViscoplasticMaterialModel, self).__init__()
        self.niv = energy_input_dim[1]
        print(
            self.niv, "Number of internal variables in the viscoplastic material model."
        )
        self.energy_function = EnergyFunction()
        self.dissipation_potential = InverseDissipationPotential(out_dim=out_dim, z_dim=z_dim, u_dim=u_dim)
        self.dt = dt  # Time step size

        # self.fnm1 = FNF1d(
        #     modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        # )
        # self.fnm2 = FNF1d(
        #     modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        # )

        self.fnm3 = FNF1d(
            modes1=modes, width=32, width_final=64, d_in=3, d_out=out_dim, n_layers=3
        )
        # self.fnm4 = FNF1d(
        #     modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        # )

    def microstructure_encoder(self, Y, n, edot_0):
        microstructure = torch.stack((Y, n, edot_0), dim=1)
        #     features1 = self.fnm1(microstructure)
        #     features2 = self.fnm2(microstructure)
        features3 = self.fnm3(microstructure)
        # features3 = microstructure
        #     features4 = self.fnm4(microstructur
        return features3

    def forward(self, e, E, Y, n, edot_0):

        stress = []
        xi = [
            torch.zeros(
                e.shape[0], self.niv, requires_grad=True, dtype=e.dtype, device=e.device
            )
        ]
        m_features1 = 1 / torch.mean(1 / E, axis=1, keepdim=True)
        m_features2 = self.microstructure_encoder(Y, n, edot_0)
        for i in tqdm.trange(0, e.shape[1]):
            s_eq, d = self.compute_energy_derivative(e[:, i], xi[i], m_features1)
            kinetics = self.compute_dissipation_derivative(-d, m_features2)
            stress.append(s_eq)
            if i < e.shape[1] - 1:
                xi.append(xi[-1] + self.dt * kinetics)
        stress = torch.stack(stress, dim=1)
        xi = torch.stack(xi, dim=1)
        return stress, xi

    def compute_energy_derivative(self, u, v, m_features):
        return self.energy_function.compute_derivative(u, v, m_features)

    def compute_dissipation_derivative(self, d, m_features):
        return self.dissipation_potential.compute_derivative(d, m_features)
