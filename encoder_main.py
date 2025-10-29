import pickle, torch, os, wandb
from tqdm import tqdm
import argparse

# Custom imports
from util import LossFunction
from m_encoder import *


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, help="Identifier for the training run")
parser.add_argument(
    "--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="data/2024-10-13_PC1D_process10_data.pkl",
    help="Path to the data file",
)
parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of training epochs"
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate for the optimizer"
)
parser.add_argument(
    "--hidden_dim", type=int, default=128, help="Hidden dimension for the autoencoder"
)
parser.add_argument(
    "--latent_dim", type=int, default=10, help="Latent dimension for the autoencoder"
)
parser.add_argument(
    "--step", type=int, default=50, help="Step size for downsampling the data"
)
parser.add_argument(
    "--n_samples", type=int, default=10, help="Number of samples to use for training"
)

args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="sci-ai",
    # Set the wandb project where this run will be logged.
    project="microstructure-encoder",
    # Track hyperparameters and run metadata.
    config=args.__dict__,
    # Name of the run
    name=args.run_id,
)

# Load a pickle file
with open(args.data_path, "rb") as f:
    data = pickle.load(f)

N = args.n_samples
step = args.step
E = torch.tensor(data["E"][:N], dtype=torch.float32).to(device)
nu = torch.tensor(data["nu"][:N], dtype=torch.float32).to(device)

loss_function = LossFunction()

ae_E = AutoEncoder(E.shape[1], args.hidden_dim, args.latent_dim).to(device)
ae_E_optimizer = torch.optim.Adam(ae_E.parameters(), lr=args.lr)
ae_E_loss_history = []

num_epochs = args.epochs
for epoch in tqdm(range(num_epochs)):
    loss = train_step(ae_E, ae_E_optimizer, E)
    ae_E_loss_history.append(loss)
    run.log(
        {
            "AE_E_Loss": loss,
            "E_epoch": epoch,
            "E_lr": ae_E_optimizer.param_groups[0]["lr"],
        }
    )
    if (epoch + 1) % 100 == 0:
        rel_error = loss_function.L2RelativeError(
            ae_E(E).unsqueeze(-1), E.unsqueeze(-1), reduction="mean"
        ).item()
        run.log({"AE_E_Relative_Error": rel_error})
        tqdm.write(
            f"AE E Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Rel_Error: {rel_error:.4f}"
        )

ae_nu = AutoEncoder(nu.shape[1], args.hidden_dim, args.latent_dim).to(device)
ae_nu_optimizer = torch.optim.Adam(ae_nu.parameters(), lr=args.lr)
ae_nu_loss_history = []

for epoch in tqdm(range(num_epochs)):
    loss = train_step(ae_nu, ae_nu_optimizer, nu)
    ae_nu_loss_history.append(loss)
    run.log(
        {
            "AE_nu_Loss": loss,
            "AE_epoch": epoch,
            "AE_lr": ae_nu_optimizer.param_groups[0]["lr"],
        }
    )
    if (epoch + 1) % 100 == 0:
        rel_error = loss_function.L2RelativeError(
            ae_nu(nu).unsqueeze(-1), nu.unsqueeze(-1), reduction="mean"
        ).item()
        run.log({"AE_nu_Relative_Error": rel_error})
        tqdm.write(
            f"AE Nu Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Rel_Error: {rel_error:.4f}"
        )

save_path = "encoder_run_{0}".format(args.run_id)
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Save scripts used in the run and model
# Scripts
os.system("cp microstructure_encoder_main.py {0}/".format(save_path))
os.system("cp m_encoder.py {0}/".format(save_path))
os.system("cp util.py {0}/".format(save_path))

# Models
torch.save(ae_E.state_dict(), "{0}/ae_E.pth".format(save_path))
torch.save(ae_nu.state_dict(), "{0}/ae_nu.pth".format(save_path))
torch.save(args.__dict__, "{0}/args.pkl".format(save_path))

run.finish()
