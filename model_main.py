import torch, argparse, wandb, os
from tqdm import tqdm
import importlib, time
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torch.optim import lr_scheduler

start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, help="Identifier for the training run")
parser.add_argument(
    "--data_path",
    type=str,
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
    "--encoder_path",
    type=str,
    default="encoder_run_1d",
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

parser.add_argument("--niv", type=int, default=1, help="Number of internal variables")

parser.add_argument(
    "--mode", type=str, default="disabled", help="Number of internal variables"
)
parser.add_argument(
    "--pca", type=int, default="True", help="Number of internal variables"
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
    mode=args.mode,
)

with open(f"{args.data_path}", "r") as f:
    content = f.read().strip()

data_files = [file.strip() for file in content.split("\n")]
print(data_files)

datasets = [
    ViscoelasticDataset(
        data_path=file,
        step=args.step,
        device=device,
        encoder=False,
    )
    for file in data_files
]
dataset = ConcatDataset(datasets)
# length = len(dataset)
# print(length)
# indices = torch.load(f"{args.indices}/dataset_indices.pth")
# trainset = Subset(dataset, indices["train_indices"])
# valset = Subset(dataset, indices["val_indices"])

length = len(dataset)
train_length, val_length = int(0.8 * length), length - int(0.8 * length)
trainset, valset = random_split(dataset, [train_length, val_length])
indices = {"train_indices": trainset.indices, "val_indices": valset.indices}
train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

loss_function = LossFunction()

encoder_input_dim = 501


if args.pca:
    print("Using PCA Encoder")
    ae_E = torch.load(f"{args.encoder_path}/ae_E.pth")
    ae_nu = torch.load(f"{args.encoder_path}/ae_nu.pth")
    ae_E.initialize_weights(args.encoder_latent_dim)
    ae_nu.initialize_weights(args.encoder_latent_dim)
else:
    print("Using AutoEncoder")
    ae_E = AutoEncoder(
        encoder_input_dim, args.encoder_hidden_dim, args.encoder_latent_dim
    ).to(device)
    ae_nu = AutoEncoder(
        encoder_input_dim, args.encoder_hidden_dim, args.encoder_latent_dim
    ).to(device)
    ae_E.load_state_dict(torch.load(f"{args.encoder_path}/ae_E.pth", weights_only=True))
    ae_nu.load_state_dict(
        torch.load(f"{args.encoder_path}/ae_nu.pth", weights_only=True)
    )


energy_input_dim = (1, args.niv, args.encoder_latent_dim * 2)
energy_hidden_dim = args.hidden_dim
dissipation_input_dim = energy_input_dim  # (p_dim, q_dim, m_dim)
dissipation_hidden_dim = args.hidden_dim

ae_E.freeze_encoder()
ae_nu.freeze_encoder()

vmm = mm.ViscoelasticMaterialModel(
    energy_input_dim,
    energy_hidden_dim,
    dissipation_input_dim,
    dissipation_hidden_dim,
    ae_E.encoder,
    ae_nu.encoder,
    dt=args.step / 5000.0,
).to(device)
optimizer = torch.optim.Adam(vmm.parameters(), lr=args.lr)
schduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# continue_training_script
# vmm.load_state_dict(torch.load("material_model_run_z7/vmm.pth"))
# optimizer.load_state_dict(torch.load("material_model_run_z7/optimizer.pth"))

# wandb.watch(vmm, log="all", log_freq=10)
epochs = args.epochs
checkpoint = 0
for epoch in tqdm(range(epochs)):
    rel_error = 0.0
    for batch_x, batch_y in train_dataloader:
        # print("epoch:", epoch)
        loss = mm.train_step(vmm, optimizer, *batch_x, batch_y)
        rel_error += loss_function.L2RelativeError(
            vmm(*batch_x)[0], batch_y, reduction="sum"
        ).item()
        # print("train", loss)
    rel_error /= len(trainset)
    tqdm.write(
        f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Rel_Error: {rel_error:.4f}"
    )

    # Validation step
    val_rel_error = 0.0
    val_loss = 0.0
    for val_batch_x, val_batch_y in val_dataloader:
        val_loss += F.mse_loss(
            vmm(*val_batch_x)[0], val_batch_y, reduction="mean"
        ).item()
        val_rel_error += loss_function.L2RelativeError(
            vmm(*val_batch_x)[0], val_batch_y, reduction="sum"
        ).item()
    val_loss /= len(val_dataloader)
    val_rel_error /= len(valset)
    # print("val", val_loss)

    schduler.step(val_loss)
    wandb.log(
        {
            "loss": loss,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "Relative_Error": rel_error,
            "val_Relative_Error": val_rel_error,
            "val_loss": val_loss,
        }
    )

    curr_time = time.time()
    time_diff = (curr_time - start_time) // (args.hrs * 3600)
    if time_diff == checkpoint:
        checkpoint += 1

        save_path = "material_model_run_{0}".format(args.run_id)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(vmm.state_dict(), "{0}/vmm.pth".format(save_path))
        torch.save(args.__dict__ | indices, "{0}/args.pkl".format(save_path))
        torch.save(optimizer.state_dict(), "{0}/optimizer.pth".format(save_path))
        os.system("cp *.py {0}/".format(save_path))
        os.system("cp *.txt {0}/".format(save_path))
