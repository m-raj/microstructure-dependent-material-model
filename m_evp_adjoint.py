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
        return F.softplus(x)


class EnergyFunction(nn.Module):
    def __init__(self, y_dim, out_dim, z_dim, u_dim, bias1=False, bias2=False):
        super(EnergyFunction, self).__init__()

        # self.picnn1 = PartiallyInputConvex(
        #     y_dim=y_dim,
        #     x_dim=out_dim,
        #     z_dim=z_dim,
        #     u_dim=u_dim,
        #     bias1=bias1,
        #     bias2=bias2,
        # )
        self.nn = nn.Sequential(
            nn.Linear(3, 50),
            CustomActivation(),
            nn.Linear(50, 1),
        )

        # self.parameter = nn.Parameter(torch.tensor(0.25))

    def forward(self, u, v, m_features):
        # print(m_features.shape, v.shape)
        # energy = 0.5 * m_features * torch.pow(u - v, 2)
        m_features = m_features.expand(*v.shape[:-1], m_features.shape[-1])
        # energy = self.picnn1(u, torch.cat((v, m_features), dim=-1))
        energy = self.nn(torch.cat((u, v, m_features), dim=-1))
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


class DissipationPotential(nn.Module):
    def __init__(self, niv, out_dim, z_dim, u_dim, bias1=False, bias2=False):
        super(DissipationPotential, self).__init__()
        self.picnn1 = PartiallyInputConvex(
            y_dim=niv, x_dim=out_dim, z_dim=z_dim, u_dim=u_dim, bias1=bias1, bias2=bias2
        )
        # self.picnn1 = PartiallyInputConvex(
        #     y_dim=1, x_dim=out_dim, z_dim=z_dim, u_dim=u_dim
        # )
        # self.dense_network = nn.Sequential(
        #     nn.Linear(10, 1000), CustomActivation(), nn.Linear(1000, 1)
        # )
        # self.parameter = nn.Parameter(0.5+torch.randn(1000)**2)

    def forward(self, q, m_features):
        # features = torch.cat((q, m_features), dim=1)
        # potential = self.dense_network(features)

        # print(m_features.shape, q.shape)

        # potential = self.parameter * m_features * torch.pow(q, 2)
        # print(m_features.shape, q.shape)
        # a, b = torch.split(m_features, m_features.shape[-1] // 2, dim=-1)
        # power = 1.2 + torch.square(b.reshape(b.shape[0], *([1] * (q.ndim - 1))))
        # a, b = torch.split(m_features, m_features.shape[-1] // 2, dim=-1)
        # power = 2.2 + torch.nn.functional.tanh(m_features)
        # print(power.min().item(), power.max().item())
        # power = torch.linspace(1.2, 2, q.shape[0])
        # power = power.reshape(q.shape[0], *([1] * (q.ndim - 1)))
        potential = self.picnn1(q, m_features)
        # parameter = self.parameter.reshape(self.parameter.shape[0], *([1] * (q.ndim - 1)))
        # potential = parameter * torch.pow(torch.abs(q), 1.5)

        # ("Power shape does not match q shape.")
        # # print(power.shape, q.shape)
        # if m_features.ndim != self.parameter.ndim:
        #     parameter = self.parameter.unsqueeze(-1)
        # else:
        #     parameter = self.parameter
        # potential = torch.square(parameter) * torch.pow(torch.abs(q), 2)

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
            y_dim=ey_dim,
            out_dim=eout_dim + niv,
            z_dim=ez_dim,
            u_dim=eu_dim,
            bias1=False,
            bias2=True,
        )
        self.dissipation_potential = DissipationPotential(
            niv=niv, out_dim=out_dim, z_dim=z_dim, u_dim=u_dim, bias1=False, bias2=False
        )
        self.dt = dt  # Time step size

        self.fnm1 = FNF1d(
            modes1=modes, width=32, width_final=64, d_in=1, d_out=eout_dim, n_layers=3
        )

        self.fnm2 = FNF1d(
            modes1=4, width=32, width_final=64, d_in=2, d_out=out_dim, n_layers=3
        )

    def microstructure_encoder(self, E, Y, n, edot_0):
        microstructure = torch.stack((1 / Y, edot_0), dim=1)
        # features1 = 1 / torch.mean(1 / E, axis=1, keepdim=True)
        features1 = self.fnm1(E.unsqueeze(1))
        features2 = self.fnm2(microstructure)
        # features3 = self.fnm3(torch.stack((Y, edot_0), dim=1))
        # features2 = torch.cat((features2, features3), dim=1)
        # features2 = features1
        return features1, features2

    def forward(self, e, E, Y, n, edot_0):
        stress = []
        xi = [torch.zeros(e.shape[0], self.niv, dtype=e.dtype, device=e.device)]
        m_features1, m_features2 = self.microstructure_encoder(E, Y, n, edot_0)
        objective = lambda u, p, q: self.energy_function(
            u, p, m_features1
        ) + self.dt * self.dissipation_potential((p - q) / self.dt, m_features2)
        for i in range(0, e.shape[1]):
            # Initialization of internal variable at the current time step using extrapolation from previous two time steps
            with torch.no_grad():
                # if i > 1:
                #     xi_dot_guess = (xi[-1] - xi[-2]) / self.dt
                # else:
                #     xi_dot_guess = torch.zeros_like(xi[-1]) + 1e-4
                # xi_guess = xi[-1]  # + self.dt * xi_dot_guess
                if i == 0:
                    diff = e[:, 1]
                elif i == e.shape[1] - 1:
                    diff = e[:, -1] - e[:, -2]
                else:
                    diff = e[:, i + 1] - e[:, i]

                xi_guess = xi[-1] + diff / 2
                # print(xi_guess.shape)
            gradient = torch.ones_like(xi_guess)
            count = 1
            error = 1
            while error > self.tol:
                if xi_guess.isnan().any():
                    # print("Time step", i)
                    assert not (
                        xi_guess.isnan().any()
                    ), "NaN detected in xi_guess during solver iterations."
                xi_guess.requires_grad_(True)
                energy = objective(e[:, i], xi_guess, xi[-1])
                gradient = torch.autograd.grad(energy.sum(), xi_guess)[0]
                # print(grad.shape, gradient.shape)
                # if (xi_guess == 0).any():
                #     print("Zero value detected in xi_guess during solver iterations.")
                if gradient.isnan().any():
                    print("Energy", energy.mean().item())
                    print(xi_guess[gradient.isnan()])
                assert not (
                    gradient.isnan().any()
                ), "NaN detected in gradient during solver iterations."
                # gradient = gradient.clip(-1, 1) / 400
                # print(torch.norm(gradient).item())
                xi_guess = (xi_guess - self.lr * gradient / count).detach()
                assert not (
                    xi_guess.isnan().any()
                ), "NaN detected in xi_guess after update during solver iterations."
                if count > 1:
                    error = torch.norm(xi_guess - xi_guess_old).item() / (
                        torch.norm(xi_guess_old).item() + 1e-8
                    )
                xi_guess_old = xi_guess.clone()
                count += 1
            xi.append(xi_guess.detach())
        xi = torch.stack(xi, dim=1)
        m_features1 = m_features1.unsqueeze(1).expand(
            e.shape[0], e.shape[1], m_features1.shape[-1]
        )
        m_features2 = m_features2.unsqueeze(1).expand(
            e.shape[0], e.shape[1], m_features2.shape[-1]
        )
        e.requires_grad_(True)
        stress = torch.autograd.grad(
            self.energy_function(e, xi[:, :-1], m_features1).sum(), e
        )[0]
        return stress, xi

    def adjoint_loss(
        self, y_true, xi=None, e=None, E=None, Y=None, n=None, edot_0=None
    ):
        with torch.no_grad():
            self.xi_dot = torch.diff(xi, n=1, dim=1) / self.dt
            # self.xi_dot[:, 0] = 2*self.xi_dot[:,1]-self.xi_dot[:,2] + 1e-7  # Handle the first time step separately
            xi = xi[:, :-1]

        m_features1, m_features2 = self.microstructure_encoder(E, Y, n, edot_0)
        m_features1 = m_features1.unsqueeze(1)
        m_features2 = m_features2.unsqueeze(1)
        # m_features1 = m_features1.expand(e.shape[0], e.shape[1], m_features1.shape[2])
        # m_features2 = m_features2.expand(
        #     xi_dot.shape[0], xi_dot.shape[1], m_features2.shape[2]
        # )
        # print(m_features1.shape, m_features2.shape, e.shape, xi.shape, xi_dot.shape)

        # print(
        #     m_features1.isnan().any(),
        #     m_features2.isnan().any(),
        #     e.isnan().any(),
        #     xi.isnan().any(),
        #     xi_dot.isnan().any(),
        # )

        f = lambda u, p: torch.square(
            torch.autograd.grad(
                self.energy_function(u, p, m_features1).sum(),
                u,
                retain_graph=True,
                create_graph=True,
            )[0]
            - y_true
        )

        g = (
            lambda u, p, q: torch.autograd.grad(
                self.energy_function(u, p, m_features1).sum(),
                p,
                retain_graph=True,
                create_graph=True,
            )[0]
            + torch.autograd.grad(
                self.dissipation_potential(q, m_features2).sum(),
                q,
                retain_graph=True,
                create_graph=True,
            )[0]
        )

        objective = lambda u, p, q, l: (f(u, p) + l * g(u, p, q))

        e.requires_grad_(True)
        self.xi_dot.requires_grad_(True)
        xi.requires_grad_(True)

        (self.Q,) = torch.autograd.grad(f(e, xi).sum(), xi)
        self.B, self.C = torch.autograd.grad(
            g(e, xi, self.xi_dot).sum(), [xi, self.xi_dot]
        )
        self.C.nan_to_num_(nan=1, posinf=1e8, neginf=-1e8)
        # C[C == 0] = 1e-4
        self.lam = torch.zeros_like(xi)
        for n in reversed(range(1, self.lam.shape[1])):
            self.lam[:, n - 1] = (
                self.lam[:, n] * (self.C[:, n] - self.B[:, n - 1] * self.dt)
                - self.Q[:, n] * self.dt
            ) / (self.C[:, n - 1] + 1e-6)
            # if lam[:, n - 1].isinf().any():
            #     print("Infinity detected in lambda computation at time step", n - 1)
            #     print(xi_dot[:, n - 1])

        # for n in reversed(range(1, lam.shape[1])):
        #     lam[:, n - 1] = (
        #         lam[:, n] * (C[:, n] - B[:, n] * self.dt) - Q[:, n] * self.dt
        #     ) / C[:, n - 1]
        #     # if lam[:, n - 1].isinf().any():
        #     #     print("Infinity detected in lambda computation at time step", n - 1)
        #     #     print(xi_dot[:, n - 1])
        self.lam.nan_to_num_(nan=0, posinf=1e8, neginf=-1e8)

        obj = objective(e, xi, self.xi_dot, self.lam)
        nan_index = torch.abs(obj).amax(dim=(1, 2)) < 1.0
        # print(
        #     lam.isnan().any(),
        #     obj.isnan(),
        #     f(e, xi).isnan().any(),
        #     g(e, xi, xi_dot).isnan().any(),
        # )
        # return C, lam, f(e, xi), g(e, xi, xi_dot), obj
        # assert not (obj.isnan().any()), "NaN detected in adjoint loss computation."

        return torch.mean(obj[nan_index])

    def compute_energy_derivative(self, u, v, m_features, **kwargs):
        return self.energy_function.compute_derivative(u, v, m_features, **kwargs)

    def compute_dissipation_derivative(self, d, m_features, **kwargs):
        return self.dissipation_potential.compute_derivative(d, m_features, **kwargs)
