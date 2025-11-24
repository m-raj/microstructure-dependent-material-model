import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnergyFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, u, u_dot, v, m_features):
        print(u.shape, u_dot.shape, v.shape, m_features.shape)
        x = torch.cat((u, u_dot, v, m_features), dim=-1)
        x = F.relu(self.fc1(x))
        energy = self.fc2(x)
        return energy.squeeze(-1)


class InverseDissipationPotential(nn.Module):
    def __init__(self, input_dim, hidden_dim, niv):
        super(InverseDissipationPotential, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, niv)

    def forward(self, p, p_dot, q, m_features):
        x = torch.cat((p, p_dot, q, m_features), dim=-1)
        x = F.relu(self.fc1(x))
        potential = self.fc2(x)
        return potential.squeeze(-1)


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
        self.sdim = energy_input_dim[0]
        self.niv = energy_input_dim[1]
        self.mdim = energy_input_dim[2]
        self.energy_function = EnergyFunction(
            self.sdim * 2 + self.niv + self.mdim, energy_hidden_dim
        )

        sdim = dissipation_input_dim[0]
        niv = dissipation_input_dim[1]
        mdim = dissipation_input_dim[2]

        self.dissipation_potential = InverseDissipationPotential(
            sdim * 2 + niv + mdim,
            dissipation_hidden_dim,
            niv,
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
            m_features = self.microstructure_encoder(E, nu)
            s = self.energy_function(e[:, i], e_dot[:, i], xi[-1], m_features)
            kinetics = self.dissipation_potential(
                e[:, i], e_dot[:, i], xi[-1], m_features
            )
            stress.append(s)
            if i < e.shape[1] - 1:
                xi.append(xi[-1] + self.dt * kinetics)
        stress = torch.stack(stress, dim=1)
        xi = torch.stack(xi, dim=1)
        return stress, xi


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
