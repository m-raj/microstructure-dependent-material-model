import torch
import glob

# torch.multiprocessing.set_start_method("spawn")
import importlib, os
from torch.utils.data import DataLoader, random_split, ConcatDataset
import lightning.pytorch as lp
import args_parser

args = args_parser.parse_args()
mm = importlib.import_module(args.material_model)
from util import *
from m_encoder import *
from lightning_script import *

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    wandb_logger = lp.loggers.WandbLogger(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="sci-ai",
        # Set the wandb project where this run will be logged.
        project="combined_encoding_training",
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
            encoder=True,
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
    trainset, valset, testset = random_split(dataset, [0.8, 0.1, 0.1])
    indices = {
        "train_indices": trainset.indices,
        "val_indices": valset.indices,
        "test_indices": testset.indices,
    }
    train_dataloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    loss_function = LossFunction()

    encoder_input_dim = 501
    folder = f"./material_model_run_{args.run_id}/"

    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(args.__dict__ | indices, "{0}/args.pkl".format(folder))
    os.system("cp *.py {0}/".format(folder))
    os.system("cp *.txt {0}/".format(folder))

    print(args.pca)
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
        lit_ae_E = LitAutoEncoder(ae_E, name="E_")
        model_checkpoint = lp.callbacks.ModelCheckpoint(
            every_n_epochs=100,
            dirpath=folder,
            filename="ae_E-{epoch:02d}-{train_rel_error:.4f}",
            mode="min",
        )
        lr_monitor = lp.callbacks.LearningRateMonitor(logging_interval="epoch")
        early_stopping = lp.callbacks.EarlyStopping(
            monitor="E_val_mse",
            patience=20,
            mode="min",
        )
        trainer_ae_E = lp.Trainer(
            max_epochs=args.encoder_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=wandb_logger,
            callbacks=[model_checkpoint, lr_monitor, early_stopping],
        )

        ae_nu = AutoEncoder(
            encoder_input_dim, args.encoder_hidden_dim, args.encoder_latent_dim
        ).to(device)
        lit_ae_nu = LitAutoEncoder(ae_nu, name="nu_")

        model_checkpoint = lp.callbacks.ModelCheckpoint(
            every_n_epochs=100,
            dirpath=folder,
            filename="ae_nu-{epoch:02d}-{train_rel_error:.4f}",
            mode="min",
        )
        lr_monitor = lp.callbacks.LearningRateMonitor(logging_interval="epoch")
        early_stopping = lp.callbacks.EarlyStopping(
            monitor="nu_val_mse",
            patience=20,
            mode="min",
        )
        trainer_ae_nu = lp.Trainer(
            max_epochs=args.encoder_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=wandb_logger,
            callbacks=[model_checkpoint, lr_monitor, early_stopping],
        )

        if args.encoder_path:
            print("Loading encoder weights from:", args.encoder_path)
            file = glob.glob(f"{args.encoder_path}/ae_E*.ckpt")
            print(file)
            lit_ae_E.load_state_dict(
                torch.load(file[0], weights_only=True)["state_dict"]
            )
            trainer_ae_E.test(lit_ae_E, test_dataloader)
            file = glob.glob(f"{args.encoder_path}/ae_nu*.ckpt")
            print(file)
            lit_ae_nu.load_state_dict(
                torch.load(file[0], weights_only=True)["state_dict"]
            )
            trainer_ae_nu.test(lit_ae_nu, test_dataloader)
            print("Encoders loader successfully.")
        else:
            print("No encoder path provided, training from scratch.")

            trainer_ae_E.fit(lit_ae_E, train_dataloader, val_dataloader)
            trainer_ae_E.test(lit_ae_E, test_dataloader)

            trainer_ae_nu.fit(lit_ae_nu, train_dataloader, val_dataloader)
            trainer_ae_nu.test(lit_ae_nu, test_dataloader)

    energy_input_dim = (1, args.niv, args.encoder_latent_dim * 2)
    energy_hidden_dim = list(
        map(lambda l: int(l.strip()), args.hidden_dim.strip().split(","))
    )
    dissipation_input_dim = energy_input_dim  # (p_dim, q_dim, m_dim)
    dissipation_hidden_dim = list(
        map(lambda l: int(l.strip()), args.hidden_dim.strip().split(","))
    )
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

    epochs = args.epochs

    model_checkpoint = lp.callbacks.ModelCheckpoint(
        every_n_epochs=100,
        dirpath=folder,
        filename="vmm-{epoch:02d}-{train_rel_error:.4f}",
        mode="min",
    )
    lr_monitor = lp.callbacks.LearningRateMonitor(logging_interval="epoch")
    early_stopping = lp.callbacks.EarlyStopping(
        monitor="vmm_val_mse",
        patience=20,
        mode="min",
    )

    lit = LitVMM(
        vmm,
        name="vmm_",
        loss_type=args.loss_type,
        lr=args.lr,
    )
    trainer = lp.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[model_checkpoint, lr_monitor],
        logger=wandb_logger,
        inference_mode=False,
    )

    trainer.fit(lit, train_dataloader, val_dataloader)

    if early_stopping.stopping_reason_message:
        print(f"Details: {early_stopping.stopping_reason_message}")

    trainer.test(lit, test_dataloader)

    wandb_logger.experiment.finish()
