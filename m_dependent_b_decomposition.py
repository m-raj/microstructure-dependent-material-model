import torch
import torch.nn as nn
import torch.nn.functional as F

# Decompsition of W and D into three parts


class ReLU2(nn.Module):
    def __init__(self):
        super(ReLU2, self).__init__()

    def forward(self, x):
        return torch.square(F.relu(x))


class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()

    def forward(self, x):
        return torch.square(x)


class EnergyFunction(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(EnergyFunction, self).__init__()

        self.E0 = nn.Sequential(
            nn.Linear(input_dim[2], hidden_dims[0]),
            ReLU2(),
            nn.Linear(hidden_dims[0], input_dim[0]),
            Square(),
        )

        self.A = nn.Sequential(
            nn.Linear(input_dim[2], hidden_dims[0]),
            ReLU2(),
            nn.Linear(hidden_dims[0], input_dim[1]),
            Square(),
        )

    def forward(self, u, v, m_features):
        energy = 1 / 2 * self.E0(m_features) * u**2 + 1 / 2 * torch.sum(
            self.A(m_features) * (u - v) ** 2, dim=-1, keepdim=True
        )

        # x = torch.cat((u, v, m_features), dim=-1)
        # for layer in self.layers:
        #     x = self.activation(layer(x))
        # energy = self.output_layer(x)
        return energy.squeeze(-1)

    def compute_derivative(self, u, v, m_features):
        u.requires_grad_(True)
        energy = self(u, v, m_features)
        grad_u, grad_v = torch.autograd.grad(
            energy.sum(), (u, v), create_graph=True, retain_graph=True
        )
        return grad_u, grad_v


class InverseDissipationPotential(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(InverseDissipationPotential, self).__init__()
        self.nu0 = nn.Sequential(
            nn.Linear(input_dim[2], hidden_dims[0]),
            ReLU2(),
            nn.Linear(hidden_dims[0], input_dim[0]),
            Square(),
        )

        self.BbyA = nn.Sequential(
            nn.Linear(input_dim[2], hidden_dims[0]),
            ReLU2(),
            nn.Linear(hidden_dims[0], input_dim[1]),
            Square(),
        )

    def forward(self, p, q, m_features):
        p.requires_grad_(True)
        potential = -1 / 2 * self.nu0(m_features) * p**2 + 1 / 2 * torch.sum(
            self.BbyA(m_features) * q**2, dim=-1, keepdim=True
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
        energy_input_dim,
        energy_hidden_dim,
        dissipation_input_dim,
        dissipation_hidden_dim,
        E_encoder,
        nu_encoder,
        dt,
    ):
        super(ViscoelasticMaterialModel, self).__init__()
        self.niv = energy_input_dim[1]
        self.energy_function = EnergyFunction(energy_input_dim, energy_hidden_dim)
        self.dissipation_potential = InverseDissipationPotential(
            dissipation_input_dim, dissipation_hidden_dim
        )
        self.dt = dt  # Time step size

        # Microstructure encoder
        self.E_encoder = E_encoder
        self.nu_encoder = nu_encoder

    def microstructure_encoder(self, E, nu):
        x = torch.cat((self.E_encoder(E), self.nu_encoder(nu)), dim=-1)
        return x

    def forward(self, e, e_dot, E, nu):
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
