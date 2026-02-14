import torch
import torch.nn as nn
import torch.nn.functional as F


class ReluSquare(nn.Module):
    def __init__(self):
        super(ReluSquare, self).__init__()

    def forward(self, x):
        return torch.square(F.relu(x))


class ConvexNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ConvexNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Parameter(torch.randn(hidden_dim, 1))
        self.activation = ReluSquare()

    def forward(self, x):
        x = self.activation((self.fc1(x)))
        W = torch.square(self.fc2)  # Ensure weights are positive
        output = torch.matmul(x, W)
        return output.squeeze(-1)


class PartiallyInputConvexLayer(nn.Module):
    def __init__(self, y_dim, u_dim, out_dim, z_dim=None, bias1=False, bias2=False):
        super(PartiallyInputConvexLayer, self).__init__()
        self.bias1 = bias1
        self.bias2 = bias2
        if self.bias2:
            self.fc1 = nn.Linear(u_dim, out_dim, bias=self.bias1)
        self.fc2 = nn.Linear(u_dim, y_dim)
        if z_dim:
            self.fc3 = nn.Linear(u_dim, z_dim)
            self.A = nn.Parameter(torch.randn(z_dim, out_dim))

        self.B = nn.Parameter(torch.randn(y_dim, out_dim))
        self.activation = ReluSquare()

    def forward(self, y, u, z=None):
        """
        Convex in y, not in u.
        """
        l1 = self.fc1(u) if self.bias2 else 0
        l2 = self.fc2(u)
        if z is not None:
            l3 = F.softplus(self.fc3(u))
            t1 = torch.einsum("...j,...j,ji->...i", z, l3, torch.square(self.A))
        t2 = torch.einsum("...j,...j,ji->...i", y, l2, self.B)

        output = t1 + t2 + l1 if z is not None else t2 + l1
        return output


class PartiallyInputConvex(nn.Module):
    def __init__(self, y_dim, x_dim, z_dim, u_dim, bias1, bias2):
        super(PartiallyInputConvex, self).__init__()
        self.fc1 = nn.Linear(x_dim, u_dim)
        self.activation = ReluSquare()

        self.picnn1 = PartiallyInputConvexLayer(
            y_dim, x_dim, z_dim, z_dim=None, bias1=True, bias2=True
        )
        self.picnn2 = PartiallyInputConvexLayer(
            y_dim, u_dim, 1, z_dim=z_dim, bias1=bias1, bias2=bias2
        )

    def forward(self, y, x):
        """
        Convex in y, not in x
        """
        u = self.activation(self.fc1(x))
        z1 = self.activation(self.picnn1(y, x))
        out = self.picnn2(y, u, z1)

        return out.squeeze(-1)


# class LORAweightsRank(nn.Module):
#     def __init__(self, input_size, hidden_size, dim1, dim2, rank):
#         super(LORAweightsRank, self).__init__()

#         assert rank > 0, "Rank must be positive"
#         assert rank <= min(
#             dim1, dim2
#         ), "Rank must be less than or equal to min(dim1, dim2)"

#         self.fc1 = nn.Parameter(torch.randn(rank, input_size, hidden_size))
#         self.fc2 = nn.Parameter(torch.randn(rank, hidden_size, dim1))

#         self.fc3 = nn.Parameter(torch.randn(rank, input_size, hidden_size))
#         self.fc4 = nn.Parameter(torch.randn(rank, hidden_size, dim2))

#         self.activation = nn.ReLU()

#     def return_weights(self, x):
#         row = torch.einsum("bi,rij->brj", x, self.fc1)
#         row = self.activation(row)
#         row = torch.einsum("brj,rjk->brk", row, self.fc2)

#         col = torch.einsum("bi,rij->brj", x, self.fc3)
#         col = self.activation(col)
#         col = torch.einsum("brj,rjk->brk", col, self.fc4)

#         lora_weights = torch.einsum("bri,brj->brij", row, col)
#         lora_weights = lora_weights.sum(dim=1)
#         return lora_weights

#     def forward(self, x, y=None):
#         lora_weights = self.return_weights(x)
#         return torch.einsum("bj, bji->bi", y, lora_weights)
