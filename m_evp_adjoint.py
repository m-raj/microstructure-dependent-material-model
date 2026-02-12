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
    def __init__(self, y_dim, out_dim, z_dim, u_dim):
        super(EnergyFunction, self).__init__()

        self.picnn1 = PartiallyInputConvex(
            y_dim=y_dim, x_dim=out_dim, z_dim=z_dim, u_dim=u_dim
        )

    def forward(self, u, v, m_features):
        features = torch.cat((v, m_features), dim=-1)
        energy = self.picnn1(u, features)
        return energy.squeeze(-1)

    def compute_derivative(
        self, u, v, m_features, retain_graph=True, create_graph=True
    ):
        assert (
            u.requires_grad
        ), "Input u must require gradients for derivative computation."
        assert (
            v.requires_grad
        ), "Input v must require gradients for derivative computation."
        energy = self(u, v, m_features)
        grad_u, grad_v = torch.autograd.grad(
            energy.sum(),
            (u, v),
            retain_graph=retain_graph,
            create_graph=create_graph,
        )
        return grad_u, grad_v


class InverseDissipationPotential(nn.Module):
    def __init__(self, niv, out_dim, z_dim, u_dim):
        super(InverseDissipationPotential, self).__init__()
        self.picnn1 = PartiallyInputConvex(
            y_dim=niv, x_dim=out_dim, z_dim=z_dim, u_dim=u_dim
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

    def compute_derivative(self, q, m_features, retain_graph=True, create_graph=True):
        assert (
            q.requires_grad == True
        ), "Input q must require gradients for derivative computation."
        potential = self(q, m_features)
        grad_q = torch.autograd.grad(
            potential.sum(), q, retain_graph=retain_graph, create_graph=create_graph
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
        tol=1e-3,
        lr=0.001,
        iter_limit=30,
        method=None,
    ):
        super(ViscoplasticMaterialModel, self).__init__()
        self.niv = niv
        self.tol = tol
        self.lr = lr
        self.iter_limit = iter_limit
        self.method = method
        print(
            self.niv, "Number of internal variables in the viscoplastic material model."
        )
        self.energy_function = EnergyFunction(
            y_dim=ey_dim, out_dim=eout_dim + niv, z_dim=ez_dim, u_dim=eu_dim
        )
        self.dissipation_potential = InverseDissipationPotential(
            niv=niv, out_dim=out_dim, z_dim=z_dim, u_dim=u_dim
        )
        self.dt = dt  # Time step size

        self.fnm1 = FNF1d(
            modes1=modes, width=32, width_final=64, d_in=1, d_out=eout_dim, n_layers=3
        )

        self.fnm2 = FNF1d(
            modes1=modes, width=32, width_final=64, d_in=3, d_out=out_dim, n_layers=3
        )

    def microstructure_encoder(self, E, Y, n, edot_0):
        microstructure = torch.stack((Y, n, edot_0), dim=1)
        features1 = self.fnm1(E.unsqueeze(1))
        features2 = self.fnm2(microstructure)
        return features1, features2

    def forward(self, e, E, Y, n, edot_0):
        stress = []
        xi = [torch.zeros(e.shape[0], self.niv, dtype=e.dtype, device=e.device)]
        m_features1, m_features2 = self.microstructure_encoder(E, Y, n, edot_0)
        e.requires_grad_(True)
        for i in tqdm.trange(0, e.shape[1]):
            xi_guess = 2 * xi[-1] - xi[-2] if i > 2 else xi[-1]
            error = float("inf")
            count = 0
            while error > self.tol and count < self.iter_limit:
                count += 1
                xi_guess.requires_grad_(True)
                s_eq, w_d = self.compute_energy_derivative(
                    e[:, i],
                    xi_guess,
                    m_features1,
                    retain_graph=True,
                    create_graph=True,
                )
                if w_d.isnan().any():
                    print(count, "wd", w_d.isnan().sum() / w_d.numel())
                with torch.no_grad():
                    xi_dot_guess = (xi_guess - xi[-1]) / self.dt
                xi_dot_guess.requires_grad_(True)
                D_d = self.compute_dissipation_derivative(
                    xi_dot_guess, m_features2, retain_graph=True, create_graph=True
                )
                if self.method == "newton":
                    w_dd = torch.autograd.grad(w_d.sum(), xi_guess, retain_graph=False)[
                        0
                    ]
                    D_dd = torch.autograd.grad(
                        D_d.sum(), xi_dot_guess, retain_graph=False
                    )[0]
                with torch.no_grad():
                    gradient = w_d + D_d
                    if self.method == "newton":
                        double_grad = w_dd + D_dd / self.dt
                        lr = 1.0 / double_grad
                    else:
                        lr = self.lr
                    xi_guess = (xi_guess - gradient * lr).detach()
                error = torch.mean(torch.abs(gradient)).item()
            if i < e.shape[1] - 1:
                xi.append(xi_guess)
            stress.append(s_eq)
        e.requires_grad_(False)
        stress = torch.stack(stress, dim=1)
        xi = torch.stack(xi, dim=1)
        return stress, xi

    def adjoint_loss(
        self, y_true, xi=None, e=None, E=None, Y=None, n=None, edot_0=None
    ):
        # print(xi.shape, e.shape, E.shape, Y.shape, n.shape, edot_0.shape)
        m_features1, m_features2 = self.microstructure_encoder(E, Y, n, edot_0)
        m_features1 = m_features1.unsqueeze(1)
        m_features2 = m_features2.unsqueeze(1)
        m_features1 = m_features1.expand(e.shape)
        xi_dot = (
            torch.diff(xi, n=1, dim=1, prepend=torch.zeros_like(xi[:, :1])) / self.dt
        )
        m_features2 = m_features2.expand(xi_dot.shape)

        e.requires_grad_(True)
        xi.requires_grad_(True)
        xi_dot.requires_grad_(True)
        w_e, w_xi = self.compute_energy_derivative(e, xi, m_features1)
        w_exi, w_xi2 = torch.autograd.grad(w_xi.sum(), [e, xi], retain_graph=True)
        d_xidot = self.compute_dissipation_derivative(xi_dot, m_features2)
        d_xidot2 = torch.autograd.grad(
            d_xidot.sum(),
            xi_dot,
            retain_graph=True,
        )[0]
        e.requires_grad_(False)
        xi = xi.detach()
        xi_dot = xi_dot.detach()
        with torch.no_grad():
            lam = torch.zeros_like(xi)
            N = xi.shape[1]
            C = -2 * (w_e - y_true) * w_exi
            B = (
                torch.diff(d_xidot2, dim=1, prepend=torch.zeros_like(d_xidot2[:, :1]))
                / self.dt
                - w_xi2
            )
            A = d_xidot2
            for i in reversed(range(1, N)):
                lam[:, i - 1] = (self.dt * C[:, i - 1] - lam[:, i] * A[:, i - 1]) / (
                    self.dt * B[:, i - 1] - A[:, i - 1]
                )

        return F.mse_loss(w_e, y_true) + torch.mean(lam * (w_xi + d_xidot))

    def compute_energy_derivative(self, u, v, m_features, **kwargs):
        return self.energy_function.compute_derivative(u, v, m_features, **kwargs)

    def compute_dissipation_derivative(self, d, m_features, **kwargs):
        return self.dissipation_potential.compute_derivative(d, m_features, **kwargs)
