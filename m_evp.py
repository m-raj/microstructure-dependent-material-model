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
    def __init__(self):
        super(InverseDissipationPotential, self).__init__()
        # self.picnn1 = PartiallyInputConvex(y_dim=1, x_dim=1, z_dim=50, u_dim=50)
        # self.picnn2 = PartiallyInputConvex(y_dim=1, x_dim=1, z_dim=50, u_dim=50)

    def forward(self, q, Y, n, edot_0):
        potential = torch.mean(
            torch.pow(torch.abs(q), n + 1) / (n + 1) * edot_0 * torch.pow(Y, -n),
            dim=1,
            keepdim=True,
        )
        return potential.squeeze(-1)

    def compute_derivative(self, q, Y, n, edot_0):

        potential = self(q, Y, n, edot_0)
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
    ):
        super(ViscoplasticMaterialModel, self).__init__()
        self.niv = energy_input_dim[1]
        self.energy_function = EnergyFunction()
        self.dissipation_potential = InverseDissipationPotential()
        self.dt = dt  # Time step size

        # self.fnm1 = FNF1d(
        #     modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        # )
        # self.fnm2 = FNF1d(
        #     modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        # )

        # self.fnm3 = FNF1d(
        #     modes1=4, width=32, width_final=64, d_in=1, d_out=1, n_layers=3
        # )
        # self.fnm4 = FNF1d(
        #     modes1=4, width=32, width_final=64, d_in=2, d_out=1, n_layers=3
        # )

    # def microstructure_encoder(self, E, nu):
    #     microstructure = torch.stack((E, nu), dim=1)
    #     features1 = self.fnm1(microstructure)
    #     features2 = self.fnm2(microstructure)
    #     features3 = self.fnm3(nu)
    #     features4 = self.fnm4(microstructure)
    #     return (features1, features3), (features2, features4)

    def forward(self, e, E, Y, n, edot_0):

        stress = []
        xi = [
            torch.zeros(
                e.shape[0], self.niv, requires_grad=True, dtype=e.dtype, device=e.device
            )
        ]
        m_features1 = 1 / torch.mean(1 / E, axis=1, keepdim=True)

        for i in range(0, e.shape[1]):
            s_eq, d = self.compute_energy_derivative(e[:, i], xi[i], m_features1)
            kinetics = self.compute_dissipation_derivative(-d, Y, n, edot_0)
            stress.append(s_eq)
            if i < e.shape[1] - 1:
                xi.append(xi[-1] + self.dt * kinetics)
        stress = torch.stack(stress, dim=1)
        xi = torch.stack(xi, dim=1)
        return stress, xi

    def compute_energy_derivative(self, u, v, m_features):
        return self.energy_function.compute_derivative(u, v, m_features)

    def compute_dissipation_derivative(self, d, Y, n, edot_0):
        return self.dissipation_potential.compute_derivative(d, Y, n, edot_0)
