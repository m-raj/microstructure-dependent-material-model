import torch, argparse, random, json
from torch.utils.data import DataLoader
from neural_network import *

parser = argparse.ArgumentParser(description="training viscoelastic model")

parser.add_argument("--project_id", type=str, help="The output path for saving.")

parser.add_argument("--run_id", type=str, help="The output path for saving.")

parser.add_argument(
    "--data_folder",
    default="../Project/Burigede_post_processed_data",
    type=str,
    help="Dataset Folder",
)

parser.add_argument("--time_step", default=0.005, type=float, help="Device")

parser.add_argument(
    "--ntrain", default=100, type=int, help="The number of training data."
)

parser.add_argument(
    "--nval", default=100, type=int, help="The number of validation data."
)

parser.add_argument(
    "--wbreadth", default=15000, type=int, help="The breath of each layer."
)

parser.add_argument(
    "--dbreadth", default=15000, type=int, help="The breath of each layer."
)

parser.add_argument(
    "--hbreadth", default=200, type=int, help="The breath of each layer."
)

parser.add_argument(
    "--hlayers", default=1, type=int, help="Number of hidden layers in Hmodel."
)

parser.add_argument(
    "--niv", default=6, type=int, help="The number of internal variables."
)

parser.add_argument(
    "--epochs", default=300, type=int, help="The number of epochs for training."
)

parser.add_argument(
    "--batch_size", default=64, type=int, help="Batch size for training"
)

parser.add_argument(
    "--start_learning_rate", default=5e-3, type=float, help="The breath of each layer."
)

parser.add_argument(
    "--end_learning_rate", default=1e-3, type=float, help="The breath of each layer."
)

parser.add_argument("--device", default="cuda:0", type=str, help="Device")

parser.add_argument("--dtype", default="float32", type=str, help="Dtype")


args = parser.parse_args()

index = [i for i in range(args.ntrain + args.nval)]
random.shuffle(index)
train_index = index[: args.ntrain]
val_index = index[args.ntrain : args.ntrain + args.nval]

config = args.__dict__
config["train_index"] = train_index
config["val_index"] = val_index

if not (os.path.isdir(args.run_id)):
    os.mkdir(args.run_id)
with open(args.run_id + "/config.txt", "w") as f:
    json.dump(config, f, indent=2)
f.close()

run = intialize_wandb(project_id=args.project_id, run_id=args.run_id, config=config)

# Dataset
dataset_files = [
    args.data_folder + "/C_vec.pt",
    args.data_folder + "/PK_vec.pt",
]

train_data = MaterialDataset(dataset_files, train_index, args.dtype, args.device)
val_data = MaterialDataset(dataset_files, val_index, args.dtype, args.device)

train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

# Initializaiton of models
model = ConstitutiveModel(
    wlayers=[6 + args.niv, args.wbreadth, 1],
    dlayers=[args.niv, args.dbreadth, 1],
    niv=args.niv,
    dt=args.time_step,
    dtype=args.dtype,
    device=args.device,
)

# Initialization of optimizer an loss function
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.start_learning_rate,
)

optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, args.epochs, args.end_learning_rate
)

loss_function = LossFunction()
print("Number of model parameters dmodel:", len(list(model.parameters())))


training_loop(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    optimizer_scheduler,
    loss_function,
    run,
)
print("Run complete")
