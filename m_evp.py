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
    def __init__(self, y_dim, out_dim, z_dim, u_dim, bias1=False, bias2=False):
        super(EnergyFunction, self).__init__()

        self.picnn1 = PartiallyInputConvex(
            y_dim=y_dim,
            x_dim=out_dim,
            z_dim=z_dim,
            u_dim=u_dim,
            bias1=bias1,
            bias2=bias2,
        )

    def forward(self, u, v, m_features):
        convex_features = torch.cat((u, v), dim=1)
        energy = self.picnn1(convex_features, m_features)
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
    def __init__(self, niv, out_dim, z_dim, u_dim, bias1=False, bias2=False):
        super(InverseDissipationPotential, self).__init__()
        self.picnn1 = PartiallyInputConvex(
            y_dim=niv, x_dim=out_dim, z_dim=z_dim, u_dim=u_dim, bias1=bias1, bias2=bias2
        )
        # self.picnn1 = PartiallyInputConvex(
        #     y_dim=1, x_dim=out_dim, z_dim=z_dim, u_dim=u_dim
        # )
        # self.dense_network = nn.Sequential(
        #     nn.Linear(10, 1000), CustomActivation(), nn.Linear(1000, 1)
        # )

    def forward(self, q, m_features):
        # features = torch.cat((q, m_features), dim=1)
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
        ey_dim=None,
        niv=None,
        eout_dim=None,
        ez_dim=None,
        eu_dim=None,
        dt=None,
        out_dim=None,
        modes=None,
        z_dim=None,
        u_dim=None,
    ):
        super(ViscoplasticMaterialModel, self).__init__()
        self.niv = niv
        print(
            self.niv, "Number of internal variables in the viscoplastic material model."
        )
        self.energy_function = EnergyFunction(
            y_dim=ey_dim + niv,
            out_dim=eout_dim,
            z_dim=ez_dim,
            u_dim=eu_dim,
            bias1=False,
            bias2=True,
        )
        self.dissipation_potential = InverseDissipationPotential(
            niv=niv, out_dim=out_dim, z_dim=z_dim, u_dim=u_dim, bias1=False, bias2=True
        )
        self.dt = dt  # Time step size

        self.fnm1 = FNF1d(
            modes1=modes, width=32, width_final=64, d_in=1, d_out=eout_dim, n_layers=3
        )

        self.fnm2 = FNF1d(
            modes1=modes, width=32, width_final=64, d_in=2, d_out=out_dim, n_layers=3
        )

    def microstructure_encoder(self, E, Y, n, edot_0):
        microstructure = torch.stack((Y, edot_0), dim=1)
        features1 = self.fnm1(E.unsqueeze(1))
        features2 = self.fnm2(microstructure)
        return features1, features2

    def forward(self, e, E, Y, n, edot_0):

        stress = []
        xi = [
            torch.zeros(
                e.shape[0], self.niv, requires_grad=True, dtype=e.dtype, device=e.device
            )
        ]
        # m_features1 = 1 / torch.mean(1 / E, axis=1, keepdim=True)
        m_features1, m_features2 = self.microstructure_encoder(E, Y, n, edot_0)
        for i in range(0, e.shape[1]):
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
