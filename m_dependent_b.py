import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyFunctionM(nn.Module):
    def __init__(self, input_dim, hidden_dim, E_encoder, nu_encoder):
        super(EnergyFunctionM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Microstructure encoder
        self.E_encoder = E_encoder
        self.nu_encoder = nu_encoder

    def microstructure_encoder(self, E, nu):
        x = torch.cat((self.E_encoder(E), self.nu_encoder(nu)), dim=-1)
        return x

    def forward(self, u, v, E, nu):
        m_features = self.microstructure_encoder(E, nu)
        x = torch.cat((u, v, m_features), dim=-1)
        x = torch.square(F.relu(self.fc1(x)))
        energy = self.fc2(x)
        return energy.squeeze(-1)

    def compute_derivative(self, u, v, E, nu):
        u.requires_grad_(True)
        energy = self(u, v, E, nu)
        grad_u, grad_v = torch.autograd.grad(
            energy.sum(), (u, v), create_graph=True, retain_graph=True
        )
        return grad_u, grad_v


class InverseDissipationPotentialM(nn.Module):
    def __init__(self, input_dim, hidden_dim, E_encoder, nu_encoder):
        super(InverseDissipationPotentialM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Microstructure encoder
        self.E_encoder = E_encoder
        self.nu_encoder = nu_encoder

    def microstructure_encoder(self, E, nu):
        x = torch.cat((self.E_encoder(E), self.nu_encoder(nu)), dim=-1)
        return x

    def forward(self, p, q, E, nu):
        p.requires_grad_(True)
        m_features = self.microstructure_encoder(E, nu)
        x = torch.cat((p, q, m_features), dim=-1)
        x = torch.square(F.relu(self.fc1(x)))
        potential = self.fc2(x)
        return potential.squeeze(-1)

    def compute_derivative(self, p, q, E, nu):
        potential = self(p, q, E, nu)
        grad_p, grad_q = torch.autograd.grad(
            potential.sum(), (p, q), create_graph=True, retain_graph=True
        )
        return grad_p, grad_q


class ViscoelasticMaterialModelM(nn.Module):
    def __init__(
        self,
        energy_input_dim,
        energy_hidden_dim,
        dissipation_input_dim,
        dissipation_hidden_dim,
        E_encoder,
        nu_encoder,
        dt=0.01,
    ):
        super(ViscoelasticMaterialModelM, self).__init__()
        self.energy_function = EnergyFunctionM(
            energy_input_dim, energy_hidden_dim, E_encoder, nu_encoder
        )
        dissipation_input_dim = sum(dissipation_input_dim)
        self.dissipation_potential = InverseDissipationPotentialM(
            dissipation_input_dim, dissipation_hidden_dim, E_encoder, nu_encoder
        )
        self.dt = dt  # Time step size

    def forward(self, e, e_dot, E, nu):
        stress = []
        xi = [
            torch.zeros(
                e.shape[0], 1, requires_grad=True, dtype=e.dtype, device=e.device
            )
        ]
        s_eq, d = self.compute_energy_derivative(e[:, 0], xi[0], E, nu)
        for i in range(1, e.shape[1]):
            s_neq, kinetics = self.compute_dissipation_derivative(
                e_dot[:, i - 1], -d, E, nu
            )
            xi.append(xi[-1] + self.dt * kinetics)
            stress.append(s_eq - s_neq)
            s_eq, d = self.compute_energy_derivative(e[:, i], xi[-1], E, nu)
        stress.append(s_eq - s_neq)
        stress = torch.stack(stress, dim=1)
        xi = torch.stack(xi, dim=1)
        return stress, xi

    def compute_energy_derivative(self, u, v, E, nu):
        return self.energy_function.compute_derivative(u, v, E, nu)

    def compute_dissipation_derivative(self, p, q, E, nu):
        return self.dissipation_potential.compute_derivative(p, q, E, nu)


def train_step(model, optimizer, e, e_dot, E, nu, s_true):
    model.train()
    optimizer.zero_grad()
    s_pred, _ = model(e, e_dot, E, nu)
    loss = F.mse_loss(s_pred, s_true)
    loss.backward()
    optimizer.step()
    return loss.item()


def prediction_step_M(model, e, e_dot, E, nu):
    model.eval()
    s_pred, xi = model(e, e_dot, E, nu)
    return s_pred, xi
