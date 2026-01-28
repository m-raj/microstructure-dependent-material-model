import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb, os, tqdm

class MyActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 1.3)
    
class MyActivation2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.nn.functional.relu(x), 2.0)
    
class MyActivation3(torch.nn.Module):
    def __init__(self, alpha=4.0, beta=1):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = self.beta**2

    def forward(self, x):
        return torch.log(1.0 + torch.exp(self.beta*x))*self.alpha/self.gamma


class MyActivation4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.01

    def forward(self, x):
        return torch.le(x, 0)*x*self.alpha + torch.gt(x, 0)*(x**2 + x*self.alpha)
    

class MyActivation5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = -0.01

    def forward(self, x):
        return self.alpha*torch.le(x, 0)*x**2 + torch.gt(x, 0)*(x**2)

    
class WeightActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x#torch.nn.functional.softplus(x)
    
class ConvexModule1(torch.nn.Module):
    def __init__(self, xin, zin, out, beta, device, dtype, layer_location, batch_normalization=False):
        super(ConvexModule1, self).__init__()
        self.Wx = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(xin, out, device=device, dtype=dtype)))
        self.layer_location = layer_location
        self.batch_normalization = batch_normalization
        if not(self.layer_location == 0):  # Not the first layer
            self.Wz = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(zin, out, device=device, dtype=dtype)))
        if not(self.layer_location == -1): # Not the last layer
            self.b = torch.nn.Parameter(torch.zeros(out, device=device, dtype=dtype))
        self.weight_activation = WeightActivation()
        self.activation = MyActivation2()

    def forward(self, x, z):
        if not(self.layer_location == 0):
            Wz = self.weight_activation(self.Wz)
        if not(self.layer_location == -1) and not(self.layer_location == 0):
            z = self.activation(torch.matmul(x, self.Wx) + torch.matmul(z, Wz) + self.b)
        elif self.layer_location == 0:
            z = self.activation(torch.matmul(x, self.Wx) + self.b)
        else:
            z = torch.matmul(x, self.Wx) + torch.matmul(z, Wz)
        return z    
    
class ConstitutiveModel(torch.nn.Module):
    def __init__(self, wlayers, dlayers, niv, dt, dtype, device):
        super(ConstitutiveModel, self).__init__()
        self.dt = dt 
        self.dtype = torch.float32 if dtype=='float32' else torch.float64
        self.device = device
        self.niv = niv
        wlayers1 = wlayers
        wlayers2 = wlayers.copy()
        self.wlayers1 = self.initialize_layers1(wlayers1, 'free_energy_potential1')
        self.dlayers = self.initialize_layers3(dlayers, 'kinetic_potential')
        # self.squish = MyActivation2(len(wlayers)-2.0)
        
    def initialize_layers1(self, layers, potential):
        # if potential == 'free_energy_potential' or potential == 'kinetic_potential':
        module = torch.nn.ModuleList()
        for i in range(len(layers)-2):
            module.append(torch.nn.Linear(layers[i], layers[i+1], device=self.device, dtype=self.dtype))
            module.append(MyActivation5())
        module.append(torch.nn.Linear(layers[-2], layers[-1], device=self.device, dtype=self.dtype))
        return module
    
    def initialize_layers3(self, layers, potential):
        # if potential == 'free_energy_potential' or potential == 'kinetic_potential':
        self.dmodule = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            loc = i if i < len(layers)-2 else -1
            zin = None if i == 0 else layers[i]
            self.dmodule.append(ConvexModule1(xin = layers[0], zin=zin, out=layers[i+1], beta=1, dtype=self.dtype, device=self.device, layer_location=loc))
    
    def free_energy_potential(self, eps, xi):
        # normalizer = torch.tensor([[4.7140, 1.2667, 1.2540, 4.7171, 1.2589, 4.7192]], dtype=self.dtype, device=self.device)
        # eps = eps*normalizer
        x = torch.cat((eps, xi), axis=1)
        for layer in self.wlayers1:
            x = layer(x)
        return x #+ self.corrector_potential(eps)

    def stress_cell(self, eps, xi):
        '''
        eps: [Batch_size, strain_components]
        xi: [Batch_size, internal_variable_components]
        
        output: [Batch_size, Stress_components]
        '''
        x = (self.free_energy_potential(eps, xi) - self.free_energy_at_zero()).sum()
        stress, df = torch.autograd.grad(x, [eps, xi], retain_graph=True, create_graph=True, materialize_grads=True)
        return stress - self.stress_at_zero(), df
    

    def dissipation_potential(self, d):

        Dstar = None
        for layer in  self.dmodule:
            Dstar = layer(d, Dstar)
        return Dstar

    def internal_variable_increment(self, d):
        '''
        eps: [Batch_size, strain_components]
        xi: [Batch_size, internal_variable_components]
        
        output: [Batch_size, Stress_components]
        '''
        Dstar = (self.dissipation_potential(d) - self.dissipation_at_zero()).sum()
        df = torch.autograd.grad(Dstar, d, retain_graph=True, create_graph=True, materialize_grads=True)[0]
        return df
    
    def dissipation_at_zero(self):
        x = torch.zeros(1, self.niv, requires_grad=False, device=self.device, dtype=self.dtype)
        return self.dissipation_potential(x)
    
    def free_energy_at_zero(self):
        eps = torch.zeros(1, 6, requires_grad=False, device=self.device, dtype=self.dtype)
        xi = torch.zeros(1, self.niv, requires_grad=False, device=self.device, dtype=self.dtype) 
        return self.free_energy_potential(eps, xi) 

    def stress_at_zero(self):
        eps = torch.zeros(1, 6, requires_grad=True, device=self.device, dtype=self.dtype)
        xi = torch.zeros(1, self.niv, requires_grad=False, device=self.device, dtype=self.dtype) 
        w = self.free_energy_potential(eps, xi).sum()
        return torch.autograd.grad(w, eps, create_graph=True)[0]
    
    def initialize_internal_variable(self, batch_size, niv):
        self.iv = [torch.zeros(batch_size, niv, device=self.device, dtype=self.dtype, requires_grad=True)]
    
    def internal_variable(self):
        return torch.stack(self.iv, axis=1)
    
    def forward(self, eps):
        batch_size, time_steps = eps.shape[0], eps.shape[1]
        self.initialize_internal_variable(batch_size, self.niv)
        shat = [] # Predicted stress
        for i in range(time_steps):
            shatn, df = self.stress_cell(eps[:,i], self.iv[i])
            increment = self.internal_variable_increment(-df)
            if i<time_steps-1:
                self.iv.append(self.iv[i] + self.dt*increment)
            shat.append(shatn)
        return torch.stack(shat, axis=1)
    
    def corrector_potential(self, eps, n=3, r0=4, beta=20.0):
        eps_tilde = eps/r0
        alpha = r0/beta/n
        t = torch.norm(eps_tilde, keepdim=True, dim=1)
        return alpha*torch.log(1.0 + torch.exp(beta*(torch.pow(t, n)-1.0)))

    def dissipation_grad_at_zero_norm(self):
        x = torch.zeros(1, self.niv, requires_grad=True, device=self.device, dtype=self.dtype)
        grad = torch.autograd.grad(self.dissipation_potential(x), x, retain_graph=True, create_graph=True)[0].square().mean()
        return grad 
    
    def potential_grad_at_zero_norm(self):
        eps = torch.zeros(1, 6, requires_grad=True, device=self.device, dtype=self.dtype)
        xi = torch.zeros(1, self.niv, requires_grad=True, device=self.device, dtype=self.dtype)
        grad = torch.autograd.grad(self.free_energy_cell(eps, xi), xi, retain_graph=True, create_graph=True)[0].square().mean()
        return grad 
    
class MaterialDataset(Dataset):
    def __init__(self, file, index, dtype, device, normalize=False, train=True):
        super(MaterialDataset, self).__init__()
        self.index = index
        self.device = device
        self.dtype = torch.float32 if dtype == 'float32' else torch.float64
        self.eps = torch.load(file[0])[self.index]
        self.stress = torch.load(file[1])[self.index]
        self.normalize = normalize
        if normalize and train:
            self.compute_normalization_constants()
        
    def assert_dtype_device(self, tensor):
        if not(tensor.dtype == self.dtype):
            if self.device:
                return tensor.to(self.dtype).to(self.device)
            else:
                return tensor.to(self.dtype)
        elif self.device:
            return tensor.to(self.device)
    
    def set_normalization_constants(self, train_data):
        self.eps_mean = train_data.eps_mean
        self.eps_S = train_data.eps_S
        self.eps_S_inv = train_data.eps_S_inv
        
        self.stress_mean = train_data.stress_mean
        self.stress_S = train_data.stress_S
        self.stress_S_inv = train_data.stress_S_inv
        
    def compute_normalization_constants(self):
        eps_std = torch.std(self.eps, dim=(0, 1)).to(self.device).to(self.dtype)
        self.eps_mean = torch.mean(self.eps, dim=(0, 1)).to(self.device).to(self.dtype)
        self.eps_S = torch.diag(eps_std)
        self.eps_S_inv = torch.linalg.inv(self.eps_S)
        
        stress_std = torch.std(self.stress, dim=(0, 1)).to(self.device).to(self.dtype)
        self.stress_mean = torch.mean(self.stress, dim=(0, 1)).to(self.device).to(self.dtype)
        self.stress_S = torch.diag(stress_std)
        self.stress_S_inv = torch.linalg.inv(self.stress_S)
        
    def __len__(self):
        return len(self.index)
    
    def normalization(self, eps, stress):
        return (eps-self.eps_mean)@self.eps_S_inv, (stress- self.stress_mean)@self.stress_S_inv
    
    def denormalize(self, **quantities):
        output = {}
        for key, value in quantities.items():
            if key == 'eps':
                output[key] = value@self.eps_S + self.eps_mean
            if key == 'stress' or key == 'shat':
                output[key] = value@self.stress_S + self.stress_mean
        return output
    
    def __getitem__(self, idx):
        eps, stress = self.eps[idx], self.stress[idx]
        eps, stress = map(self.assert_dtype_device, [eps, stress])
        if self.normalize:
            eps, stress = self.normalization(eps, stress)
        eps.requires_grad = True
        return self.index[idx], eps, stress
    
        
class LossFunction(torch.nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        
    def L2NormSquared(self, x):
        return torch.mean(torch.sum(torch.square(x), dim=2), dim=1)
    
    def L2RelativeErrorSquared(self, x, y, reduction='mean'):
        error = x - y
        rel_error = self.L2NormSquared(error)/self.L2NormSquared(x)
        if reduction == 'mean':
            return torch.mean(rel_error)
        elif not(reduction):
            return rel_error
        else:
            print('Not a valid reduction method: ' + reduction)
            
    def L2RelativeError(self, x, y, reduction=None):
        error = x - y
        rel_error = torch.sqrt(self.L2NormSquared(error)/self.L2NormSquared(x))
        if reduction == 'mean':
            return torch.mean(rel_error)
        elif not(reduction):
            return rel_error
        else:
            print('Not a valid reduction method: ' + reduction)
            
    def L2ErrorSquared(self, x, y, reduction=None):
        error = self.L2NormSquared(x - y)
        if not(reduction):
            return error
        elif reduction=='mean':
            return torch.mean(error)
        elif reduction=='sum':
            return torch.sum(error)
        
    def L2Error(self, x, y, reduction=None):
        error = torch.sqrt(self.L2NormSquared(x, y))
        if not(reduction):
            return error
        elif reduction=='mean':
            return torch.mean(error)
        elif reduction=='sum':
            return torch.sum(error)
        
    def forward(self, x, y):
        return self.L2ErrorSquared(x, y, reduction='mean')
    
    def weighted_relative_error(self, x, y):
        error = x - y
        num = torch.mean(torch.square(error), dim=1)
        den = torch.mean(torch.square(x), dim=1)
        return torch.mean(torch.mean(num/(den+1.0e-6), dim=1), dim=0)
        
def train_step(step_id, nnmodel, eps, stress, optimizer, loss_function, denormalize=None):
    shat = nnmodel(eps)
    loss = loss_function.L2RelativeErrorSquared(stress, shat)
    loss.backward()
    torch.nn.utils.clip_grad_value_(nnmodel.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        if denormalize:
            dic =  denormalize(stress=stress, shat=shat)
            stress = dic['stress']
            shat = dic['shat']
        rel_error = loss_function.L2RelativeError(stress, shat, reduction='mean')
    return {'LossFunctionTrain': loss.item(),
            'L2RelativeErrorTrain': rel_error.item(),
            'LearningRate': optimizer.param_groups[0]['lr'],
}
    
def validation_step(step_id, nnmodel, eps, stress, loss_function, denormalize=None):
    # with torch.no_grad():
    shat = nnmodel(eps)
    loss = loss_function.L2RelativeErrorSquared(stress, shat)
    if denormalize:
        dic =  denormalize(stress=stress, shat=shat)
        stress = dic['stress']
        shat = dic['shat']
    rel_error = loss_function.L2RelativeError(stress, shat, reduction='mean')
    return {'LossFunctionVal': loss.item(),
            'L2RelativeErrorVal': rel_error.item()}



def predict(nnmodel, dataloader, denormalize=None):
    shat = []
    eps = []
    stress = []
    iv = []
    ids = []
    for i, (idx, epsi, stressi) in enumerate(dataloader):
        shati = nnmodel(epsi)
        ivi = nnmodel.internal_variable()
        if denormalize:
            dic = denormalize(eps=epsi, shat=shati, stress=stressi)
            epsi = dic['eps']
            shati = dic['shat']
            stressi = dic['stress']
        shat.append(shati)
        eps.append(epsi)
        stress.append(stressi)
        iv.append(ivi)
        ids.append(idx)
        break
    shat, eps, stress, iv, ids = map(lambda x: torch.cat(x, axis=0), [shat, eps, stress, iv, ids])
    return {'shat': shat, 'eps': eps, 'stress': stress, 'iv': iv, 'id': ids}
               
def save_model_and_prediction(run, nnmodel, train_dataloader, val_dataloader=None, denormalize=None):
    if not(os.path.isdir(run.name)):
        os.mkdir(run.name)
    torch.save(nnmodel.state_dict(), run.name + "/constituitive_model.pt")
    prediction = predict(nnmodel, train_dataloader, denormalize)
    torch.save(prediction, run.name + '/train_prediction.pt')
    prediction = predict(nnmodel, val_dataloader, denormalize)
    torch.save(prediction, run.name + '/val_prediction.pt')
      
def training_loop(nnmodel, train_dataloader, val_dataloader, optimizer, scheduler, loss_function, run, denormalize):
    for epoch in range(run.config['n_epochs']):
        if not(epoch%10):
            val_loss = 0
            val_rel_error = 0
            for i, (idx, eps, stress) in enumerate(val_dataloader):
                val_log = validation_step(i, nnmodel, eps, stress, loss_function, denormalize=denormalize)
                # print(i, val_log)
                val_loss += val_log['LossFunctionVal']
                val_rel_error += val_log['L2RelativeErrorVal']
            val_log = {'LossFunctionVal': val_loss/(i+1), 'L2RelativeErrorVal': val_rel_error/(i+1)}
        
        for i, (idx, eps, stress) in enumerate(train_dataloader):
            # pre_conditioner(nnmodel, 50, 100)
            train_log = train_step(i, nnmodel, eps, stress, optimizer, loss_function, denormalize=denormalize)
            # if not(epoch%10):
                # print(i, train_log)
            train_log.update(val_log)
            wandb.log(train_log)
        scheduler.step()

        
            
        if not(epoch%10) and epoch>0:
            save_model_and_prediction(run, nnmodel, train_dataloader, val_dataloader, denormalize)
    
def intialize_wandb(
    project_id,
    run_id,
    time_step,
    wmodel_architecture,
    dmodel_architecture,
    number_of_internal_variables,
    lr,
    scheduler_factor,
    scheduler_patience,
    dataset_file,
    train_index,
    val_index,
    batch_size,
    manual_seed,
    dtype,
    device,
    n_epochs
):
    wandb.login()
    run = wandb.init(
        project=project_id,
        name=run_id,
        config={
            "time_step": time_step,
            "wmodel_architecture": wmodel_architecture,
            "dmodel_architecture": dmodel_architecture,
            "number_of_internal_variables": number_of_internal_variables,
            "initial_learning_rate": lr,
            "scheduler_factor": scheduler_factor,
            "scheduler_patience": scheduler_patience,
            "dataset_file": dataset_file,
            "train_index": train_index,
            "val_index": val_index,
            "batch_size": batch_size,
            "manual_seed": manual_seed,
            "dtype": dtype,
            "device": device,
            "n_epochs": n_epochs,
            "train_dataset_size": len(train_index),
            "test_dataset_size": len(val_index)
        }
    )
    return run


    
        

            
            
            
    
        
            
            
            
            
        
        
            
        
        
