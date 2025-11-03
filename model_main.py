import torch, argparse, wandb, os
from tqdm import tqdm
import importlib, time
from torch.utils.data import DataLoader, random_split

start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, help="Identifier for the training run")
parser.add_argument(
    "--data_path",
    type=str,
    default="data/2024-10-13_PC1D_process10_data.pkl",
    help="Path to the dataset",
)
parser.add_argument(
    "--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)"
)
parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of training epochs"
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate for the optimizer"
)
parser.add_argument(
    "--hidden_dim", type=int, default=20, help="Hidden dimension for the autoencoder"
)

parser.add_argument(
    "--encoder_hidden_dim",
    type=int,
    default=128,
    help="Hidden dimension for the autoencoder",
)

parser.add_argument(
    "--encoder_latent_dim",
    type=int,
    default=10,
    help="Latent dimension for the autoencoder",
)

parser.add_argument(
    "--step", type=int, default=50, help="Step size for downsampling the data"
)
parser.add_argument(
    "--n_samples", type=int, default=100, help="Number of samples to use for training"
)
parser.add_argument(
    "--encoder_path",
    type=str,
    default="encoder_run_2",
    help="Path to the encoder model",
)

parser.add_argument(
    "--material_model", type=str, default="m_dependent_b", help="Material model file"
)

parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)

parser.add_argument(
    "--hrs", type=float, default=1.0, help="Maximum training time in hours"
)

args = parser.parse_args()

mm = importlib.import_module(args.material_model)
from util import *
from m_encoder import *

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="sci-ai",
    # Set the wandb project where this run will be logged.
    project="microstructure-dependent-potential-learning",
    # Track hyperparameters and run metadata.
    config=args.__dict__,
    # Name of the run
    name=args.run_id,
)


dataset = ViscoelasticDataset(
    data_path=args.data_path, N=args.n_samples, step=args.step, device=args.device
)
length = len(dataset)
train_length, val_length = int(0.8 * length), length - int(0.8 * length)

trainset, valset = random_split(dataset, [train_length, val_length])
indices = {"train_indices": trainset.indices, "val_indices": valset.indices}
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

loss_function = LossFunction()

encoder_input_dim = 501

ae_E = AutoEncoder(
    encoder_input_dim, args.encoder_hidden_dim, args.encoder_latent_dim
).to(device)
ae_nu = AutoEncoder(
    encoder_input_dim, args.encoder_hidden_dim, args.encoder_latent_dim
).to(device)

ae_E.load_state_dict(torch.load(f"{args.encoder_path}/ae_E.pth", weights_only=True))
ae_nu.load_state_dict(torch.load(f"{args.encoder_path}/ae_nu.pth", weights_only=True))

energy_input_dim = (1, 1, args.encoder_latent_dim * 2)
energy_hidden_dim = args.hidden_dim
dissipation_input_dim = (1, 1, args.encoder_latent_dim * 2)  # (p_dim, q_dim, m_dim)
dissipation_hidden_dim = args.hidden_dim

ae_E.freeze_encoder()
ae_nu.freeze_encoder()

vmm = mm.ViscoelasticMaterialModelM(
    energy_input_dim,
    energy_hidden_dim,
    dissipation_input_dim,
    dissipation_hidden_dim,
    ae_E.encoder,
    ae_nu.encoder,
).to(device)
optimizer = torch.optim.Adam(vmm.parameters(), lr=args.lr)
loss_history = []

# continue_training_script
vmm.load_state_dict(torch.load("material_model_run_d4/vmm.pth"))

# wandb.watch(vmm, log="all", log_freq=10)
epochs = args.epochs
for epoch in tqdm(range(epochs)):
    for batch_x, batch_y in dataloader:
        # print("epoch:", epoch)
        loss = mm.train_step(vmm, optimizer, *batch_x, batch_y)
    loss_history.append(loss)
    wandb.log({"loss": loss, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})
    rel_error = loss_function.L2RelativeError(
        vmm(*batch_x)[0], batch_y, reduction="mean"
    ).item()
    wandb.log({"Relative_Error": rel_error})
    tqdm.write(
        f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Rel_Error: {rel_error:.4f}"
    )

    # Validation step
    val_rel_error = 0.0
    for val_batch_x, val_batch_y in val_dataloader:
        val_rel_error += loss_function.L2RelativeError(
            vmm(*val_batch_x)[0], val_batch_y, reduction="mean"
        ).item()
    val_rel_error /= len(valset) / val_dataloader.batch_size
    wandb.log({"val_Relative_Error": val_rel_error})

    curr_time = time.time()
    time_diff = (curr_time - start_time) / 60.0
    if time_diff > args.hrs * 60 - 20:
        break


save_path = "material_model_run_{0}".format(args.run_id)

if not os.path.exists(save_path):
    os.makedirs(save_path)


torch.save(vmm.state_dict(), "{0}/vmm.pth".format(save_path))
torch.save(args.__dict__ | indices, "{0}/args.pkl".format(save_path))
torch.save(optimizer.state_dict(), "{0}/optimizer.pth".format(save_path))

os.system("cp main.py {0}/".format(save_path))
os.system("cp m_dependent_b.py {0}/".format(save_path))
os.system("cp m_encoder.py {0}/".format(save_path))
