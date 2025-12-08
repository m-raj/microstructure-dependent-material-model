import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_id", type=str, help="Identifier for the training run")
    parser.add_argument(
        "--mode", type=str, default="disabled", help="Number of internal variables"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (cpu or cuda)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--pca", action="store_true", help="Number of internal variables"
    )
    parser.add_argument(
        "--encoder_path",
        type=str,
        default=None,
        help="Path to the encoder model",
    )
    parser.add_argument(
        "--encoder_latent_dim",
        type=int,
        default=10,
        help="Latent dimension for the autoencoder",
    )
    parser.add_argument(
        "--encoder_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for the autoencoder",
    )
    parser.add_argument(
        "--encoder_epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--encoder_batch_size", type=int, default=32, help="Batch size of the dataset"
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=1e-3, help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--material_model",
        type=str,
        default="m_dependent_b",
        help="Material model file",
    )
    parser.add_argument(
        "--hidden_dim", type=str, default="128", help="Hidden dimensions"
    )
    parser.add_argument(
        "--step", type=int, default=50, help="Step size for downsampling the data"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs"
    )

    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )

    parser.add_argument(
        "--niv", type=int, default=1, help="Number of internal variables"
    )

    return parser.parse_args()
