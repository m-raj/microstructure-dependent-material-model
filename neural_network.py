import torch
from torch.utils.data import Dataset
import wandb, os


def vectomat(vec, device, dtype):
    Mat = torch.zeros(vec.shape[0], 3, 3, device=device, dtype=dtype)
    Mat[:, 0, 0] = vec[:, 0]
    Mat[:, 0, 1] = vec[:, 1]
    Mat[:, 1, 0] = vec[:, 1]
    Mat[:, 0, 2] = vec[:, 2]
    Mat[:, 2, 0] = vec[:, 2]
    Mat[:, 1, 1] = vec[:, 3]
    Mat[:, 1, 2] = vec[:, 4]
    Mat[:, 2, 1] = vec[:, 4]
    Mat[:, 2, 2] = vec[:, 5]
    return Mat


def mattovec(Mat, device, dtype):
    vec = torch.zeros(Mat.shape[0], 6, device=device, dtype=dtype)
    vec[:, 0] = Mat[:, 0, 0]
    vec[:, 1] = Mat[:, 0, 1]
    vec[:, 2] = Mat[:, 0, 2]
    vec[:, 3] = Mat[:, 1, 1]
    vec[:, 4] = Mat[:, 1, 2]
    vec[:, 5] = Mat[:, 2, 2]
    return vec


class ReLUSquared(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2.0)


class SoftReLUSquared(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(1 + torch.exp(torch.square(x)))


class WeightActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)


class ConvexLayer(torch.nn.Module):
    def __init__(self, dimin, dimout, device, dtype, bias=True):
        super().__init__()
        k = 1.0 / torch.sqrt(torch.tensor(dimin))
        self.weight = torch.nn.Parameter(
            torch.nn.init.uniform_(
                torch.empty(dimin, dimout, device=device, dtype=dtype), -k, k
            )
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.empty(dimout, device=device, dtype=dtype), -k, k
                )
            )
        else:
            self.bias = 0
        self.weight_activation = WeightActivation()

    def forward(self, x):
        weight = self.weight_activation(self.weight)
        return torch.matmul(x, weight) + self.bias


class ConstitutiveModel(torch.nn.Module):
    def __init__(self, wlayers, dlayers, niv, dt, dtype, device):
        super(ConstitutiveModel, self).__init__()
        self.dt = dt
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = device
        self.niv = niv
        self.wlayers1 = self.initialize_wlayers(wlayers)
        self.dlayers = self.initialize_dlayers(dlayers)

    def initialize_wlayers(self, layers):
        module = torch.nn.ModuleList()
        for i in range(len(layers) - 2):
            module.append(
                torch.nn.Linear(
                    layers[i], layers[i + 1], device=self.device, dtype=self.dtype
                )
            )
            module.append(ReLUSquared())
        module.append(
            torch.nn.Linear(
                layers[-2], layers[-1], device=self.device, dtype=self.dtype
            )
        )
        return module

    def initialize_dlayers(self, layers):
        module = torch.nn.ModuleList()
        for i in range(len(layers) - 2):
            module.append(
                torch.nn.Linear(
                    layers[i], layers[i + 1], device=self.device, dtype=self.dtype
                )
            )
            module.append(ReLUSquared())
        module.append(
            ConvexLayer(
                layers[-2], layers[-1], device=self.device, dtype=self.dtype, bias=False
            )
        )
        return module

    def compute_determinant(self, eps):
        E0 = torch.tensor([1, 0, 0, 0, 0, 0], dtype=self.dtype, device=self.device)
        E1 = torch.tensor([0, 1, 0, 0, 0, 0], dtype=self.dtype, device=self.device)
        E2 = torch.tensor([0, 0, 1, 0, 0, 0], dtype=self.dtype, device=self.device)
        E3 = torch.tensor([0, 0, 0, 1, 0, 0], dtype=self.dtype, device=self.device)
        E4 = torch.tensor([0, 0, 0, 0, 1, 0], dtype=self.dtype, device=self.device)
        E5 = torch.tensor([0, 0, 0, 0, 0, 1], dtype=self.dtype, device=self.device)

        P = torch.stack((E0, E1, E2, E2, E4, E5), axis=0)
        Q = torch.stack((E3, E4, E1, E2, E4, E1), axis=0)
        R = torch.stack((E5, E2, E4, -E3, -E0, -E1), axis=0)

        det = ((eps @ P.T) * (eps @ Q.T) * (eps @ R.T)).sum(-1, keepdim=True)
        return det

    # def compute_g(self, eps):
    #     det = self.compute_determinant(eps)
    #     g = eps / torch.pow(det, 1 / 3)
    #     return g, det

    def free_energy_potential(self, eps, xi):
        eye = torch.tensor([[1, 0, 0, 1, 0, 1]], dtype=self.dtype, device=self.device)
        eps= eps - eye
        x = torch.cat((eps, xi), axis=1)
        for layer in self.wlayers1:
            x = layer(x)
        return x

    def stress_cell(self, eps, xi):
        # g, det = self.compute_g(eps)
        # corrector = self.G2(g, n=3, r0=3.5, beta=4)
        # eps_inv = mattovec(torch.linalg.inv(vectomat(eps, self.device, self.dtype)), self.device, self.dtype)
        x = self.free_energy_potential(eps, xi).sum()
        stress, df = torch.autograd.grad(
            x, [eps, xi], retain_graph=True, create_graph=True, materialize_grads=True
        )
        # stress = stress + corrector
        # weights = torch.tensor(
        #     [[1, 2, 2, 1, 2, 1]], dtype=self.dtype, device=self.device
        # )
        # stress = (stress - ((stress * eps) @ weights.T) * eps_inv / 3.0) / (
        #     torch.pow(det, 1 / 3)
        # )
        return stress, df

    def dissipation_potential(self, d):
        Dstar = self.dlayers[0](d)
        for i in range(1, len(self.dlayers)):
            Dstar = self.dlayers[i](Dstar)
        return Dstar

    def internal_variable_increment(self, d):
        Dstar = self.dissipation_potential(d).sum()
        df = torch.autograd.grad(
            Dstar, d, retain_graph=True, create_graph=True, materialize_grads=True
        )[0]
        return df

    def dissipation_at_zero(self):
        x = torch.zeros(
            1, self.niv, requires_grad=False, device=self.device, dtype=self.dtype
        )
        return self.dissipation_potential(x)

    def free_energy_at_eye(self):
        eps = torch.tensor([[1, 0, 0, 1, 0, 1]], dtype=self.dtype, device=self.device)
        xi = torch.zeros(
            1, self.niv, requires_grad=False, device=self.device, dtype=self.dtype
        )
        return self.free_energy_potential(eps, xi)

    def initialize_internal_variable(self, batch_size, niv):
        self.iv = [
            torch.zeros(
                batch_size,
                niv,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
        ]

    def internal_variable(self):
        return torch.stack(self.iv, axis=1)

    def forward(self, eps):
        batch_size, time_steps = eps.shape[0], eps.shape[1]
        self.initialize_internal_variable(batch_size, self.niv)
        shat = []  # Predicted stress
        for i in range(time_steps):
            shatn, df = self.stress_cell(eps[:, i], self.iv[i])
            increment = self.internal_variable_increment(-df)
            if i < time_steps - 1:
                self.iv.append(self.iv[i] + self.dt * increment)
            shat.append(shatn)
        return torch.stack(shat, axis=1)

    # def G2(self, g, n=3, r0=12, beta=4.0):
    #     g_norm = torch.linalg.norm(g, dim=-1, keepdim=True)
    #     t = g_norm/r0
    #     alpha = r0/beta/n
    #     return alpha*beta*n*g_norm**(n-2)*g/(r0**n*(1 + torch.exp(-beta*(torch.pow(t, n) - 1.0))))

    def dissipation_grad_at_zero_norm(self):
        x = torch.zeros(
            1, self.niv, requires_grad=True, device=self.device, dtype=self.dtype
        )
        grad = (
            torch.autograd.grad(
                self.dissipation_potential(x), x, retain_graph=True, create_graph=True
            )[0]
            .square()
            .mean()
        )
        return grad

    def potential_grad_at_zero_norm(self):
        eps = torch.zeros(
            1, 6, requires_grad=True, device=self.device, dtype=self.dtype
        )
        xi = torch.zeros(
            1, self.niv, requires_grad=True, device=self.device, dtype=self.dtype
        )
        grad = (
            torch.autograd.grad(
                self.free_energy_cell(eps, xi), xi, retain_graph=True, create_graph=True
            )[0]
            .square()
            .mean()
        )
        return grad


class MaterialDataset(Dataset):
    def __init__(self, file, index, dtype, device):
        super(MaterialDataset, self).__init__()
        self.index = index
        self.device = device
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.eps = torch.load(file[0])[self.index]
        self.stress = torch.load(file[1])[self.index]
        # self.deteps = torch.load(file[2])[self.index]
        # self.s1 = torch.load(file[3])[self.index]

    def assert_dtype_device(self, tensor):
        if not (tensor.dtype == self.dtype):
            if self.device:
                return tensor.to(self.dtype).to(self.device)
            else:
                return tensor.to(self.dtype)
        elif self.device:
            return tensor.to(self.device)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        eps, stress = self.eps[idx], self.stress[idx]
        eps, stress = map(self.assert_dtype_device, [eps, stress])
        # deteps, s1 = map(self.assert_dtype_device, [self.deteps[idx], self.s1[idx]])
        eps.requires_grad = True
        return self.index[idx], eps, stress


class LossFunction(torch.nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def L2NormSquared(self, x):
        return torch.mean(torch.sum(torch.square(x), dim=2), dim=1)

    def L2RelativeErrorSquared(self, x, y, reduction="mean"):
        error = x - y
        rel_error = self.L2NormSquared(error) / self.L2NormSquared(x)
        if reduction == "mean":
            return torch.mean(rel_error)
        elif not (reduction):
            return rel_error
        else:
            print("Not a valid reduction method: " + reduction)

    def L2RelativeError(self, x, y, reduction=None):
        error = x - y
        rel_error = torch.sqrt(self.L2NormSquared(error) / self.L2NormSquared(x))
        if reduction == "mean":
            return torch.mean(rel_error)
        elif not (reduction):
            return rel_error
        else:
            print("Not a valid reduction method: " + reduction)

    def L2ErrorSquared(self, x, y, reduction=None):
        error = self.L2NormSquared(x - y)
        if not (reduction):
            return error
        elif reduction == "mean":
            return torch.mean(error)
        elif reduction == "sum":
            return torch.sum(error)

    def L2Error(self, x, y, reduction=None):
        error = torch.sqrt(self.L2NormSquared(x, y))
        if not (reduction):
            return error
        elif reduction == "mean":
            return torch.mean(error)
        elif reduction == "sum":
            return torch.sum(error)

    def forward(self, x, y):
        return self.L2ErrorSquared(x, y, reduction="mean")

    def L2WeightedRelativeError(self, x, y):
        error = x - y
        num = torch.mean(torch.square(error), dim=1)
        den = torch.mean(torch.square(x), dim=1)
        return torch.mean(torch.mean(num / (den + 1.0e-6), dim=1), dim=0)


def train_step(model, eps, stress, optimizer, loss_function):
    shat = model(eps)
    loss = loss_function.L2RelativeErrorSquared(stress, shat)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        rel_error = loss_function.L2RelativeError(stress, shat, reduction="mean")
    return {
        "SLossFunctionTrain": loss.item(),
        "SL2RelativeErrorTrain": rel_error.item(),
        "SLearningRate": optimizer.param_groups[0]["lr"],
    }


def validation_step(model, eps, stress, loss_function):

    shat = model(eps)
    devloss = loss_function.L2RelativeErrorSquared(stress, shat)

    rel_error = loss_function.L2RelativeError(stress, shat, reduction="mean")

    return {
        "SLossFunctionVal": devloss.item(),
        "SL2RelativeErrorVal": rel_error.item(),
    }


def predict(model, dataloader, count=0):
    shat = []
    eps = []
    iv = []
    ids = []
    stress = []
    for i, (idx, epsi, stressi) in enumerate(dataloader):
        if i > count:
            break
        shati = model(epsi)
        ivi = model.internal_variable()

        shat.append(shati.cpu().detach())
        stress.append(stressi.cpu().detach())
        iv.append(ivi.cpu().detach())
        ids.append(idx.cpu().detach())
        torch.cuda.empty_cache()

    shat, eps, iv, ids, stress = map(
        lambda x: torch.cat(x, axis=0),
        [shat, eps, iv, ids, stress],
    )
    return {"shat": shat, "eps": eps, "iv": iv, "id": ids, "stress": stress}


def save_model_and_prediction(run, model, train_dataloader, val_dataloader=None):
    if not (os.path.isdir(run.name)):
        os.mkdir(run.name)
    torch.save(model.state_dict(), run.name + "/constituitive_model.pt")
    prediction = predict(model, train_dataloader)
    torch.save(prediction, run.name + "/train_prediction.pt")
    prediction = predict(model, val_dataloader)
    torch.save(prediction, run.name + "/val_prediction.pt")


def training_loop(
    model, train_dataloader, val_dataloader, optimizer, scheduler, loss_function, run
):
    for epoch in range(run.config["epochs"]):
        if not (epoch % 2):
            val_sloss = 0
            val_srelerror = 0
            for i, (idx, eps, stress) in enumerate(val_dataloader):
                val_log = validation_step(model, eps, stress, loss_function)
                val_sloss += val_log["SLossFunctionVal"]
                val_srelerror += val_log["SL2RelativeErrorVal"]
            val_log = {
                "SLossFunctionVal": val_sloss / (i + 1),
                "SL2RelativeErrorVal": val_srelerror / (i + 1),
            }
            print("val_log", val_log)

        for i, (_, eps, stress) in enumerate(train_dataloader):
            train_log = train_step(model, eps, stress, optimizer, loss_function)
            train_log.update(val_log)
            wandb.log(train_log)
        print("train_log", train_log)
        scheduler[0].step()

        if not (epoch % 10):
            save_model_and_prediction(run, model, train_dataloader, val_dataloader)


def intialize_wandb(project_id, run_id, config):
    wandb.login()
    run = wandb.init(project=project_id, name=run_id, config=config)
    return run
