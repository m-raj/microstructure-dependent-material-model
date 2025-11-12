import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnergyFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, u, v):
        x = torch.cat((u, v), dim=-1)
        x = F.relu(self.fc1(x)) ** 2
        energy = self.fc2(x)
        return energy.squeeze(-1)

    def compute_derivative(self, u, v):
        u.requires_grad_(True)
        energy = self(u, v)
        grad_u, grad_v = torch.autograd.grad(
            energy.sum(), (u, v), create_graph=True, retain_graph=True
        )
        return grad_u, grad_v


class InverseDissipationPotential(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InverseDissipationPotential, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, p, q):
        p.requires_grad_(True)
        x = torch.cat((p, q), dim=-1)
        x = F.relu(self.fc1(x)) ** 2
        potential = self.fc2(x)
        return potential.squeeze(-1)

    def compute_derivative(self, p, q):
        potential = self(p, q)
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
        E_encoder=None,
        nu_encoder=None,
        dt=0.01,
    ):
        super(ViscoelasticMaterialModel, self).__init__()
        energy_input_dim = energy_input_dim[0] + energy_input_dim[1]
        self.energy_function = EnergyFunction(energy_input_dim, energy_hidden_dim)
        dissipation_input_dim = dissipation_input_dim[0] + dissipation_input_dim[1]
        self.dissipation_potential = InverseDissipationPotential(
            dissipation_input_dim, dissipation_hidden_dim
        )
        self.dt = dt  # Time step size

    def forward(self, e, e_dot):
        stress = []
        xi = [torch.zeros_like(torch.zeros(e.shape[0], 1), requires_grad=True)]
        s_eq, d = self.compute_energy_derivative(e[:, 0], xi[0])
        for i in range(1, e.shape[1]):
            s_neq, kinetics = self.compute_dissipation_derivative(e_dot[:, i - 1], -d)
            xi.append(xi[-1] + self.dt * kinetics)
            stress.append(s_eq - s_neq)
            s_eq, d = self.compute_energy_derivative(e[:, i], xi[-1])
        stress.append(s_eq - s_neq)
        stress = torch.stack(stress, dim=1)
        xi = torch.stack(xi, dim=1)
        return stress, xi

    def compute_energy_derivative(self, u, v):
        return self.energy_function.compute_derivative(u, v)

    def compute_dissipation_derivative(self, p, q):
        return self.dissipation_potential.compute_derivative(p, q)


def train_step(model, optimizer, e, e_dot, s_true):
    model.train()
    optimizer.zero_grad()
    s_pred, _ = model(e, e_dot)
    loss = F.mse_loss(s_pred, s_true)
    loss.backward()
    optimizer.step()
    return loss.item()


def prediction_step(model, e, e_dot, E, nu):
    model.eval()
    s_pred, xi = model(e, e_dot, E, nu)
    return s_pred, xi
