import pickle, torch, os, wandb
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from util import LossFunction, ViscoelasticDataset
from m_encoder import PCAEncoder

from sklearn.decomposition import PCA
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, help="Identifier for the training run")
parser.add_argument("--features", type=int, default=20, help="Number of PCA features")
parser.add_argument(
    "--data_files", type=str, default="", help="Comma separated data files"
)
args = parser.parse_args()

folder = f"pca_encoder_{args.run_id}"

data_files = [file.strip() for file in args.data_files.split(",")]
n_features = args.features

datasets = [
    ViscoelasticDataset(
        data_path=file,
        step=50,
        device="cpu",
        encoder=True,
    )
    for file in data_files
]

loss_function = LossFunction()
dataset = ConcatDataset(datasets)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

pca_E = PCA(n_components=60)
pca_nu = PCA(n_components=60)

E_samples = []
nu_samples = []
for idx, (E, nu) in enumerate(dataloader):
    E_samples.append(E.numpy())
    nu_samples.append(nu.numpy())
E_samples = np.concatenate(E_samples, axis=0)
nu_samples = np.concatenate(nu_samples, axis=0)

train_size = int(0.8 * E_samples.shape[0])
test_size = E_samples.shape[0] - train_size
train_E_samples = E_samples[:train_size]
train_nu_samples = nu_samples[:train_size]

test_E_samples = E_samples[train_size:]
test_nu_samples = nu_samples[train_size:]

pca_E.fit(train_E_samples)
pca_nu.fit(train_nu_samples)

E_encoder = PCAEncoder(
    501,
    n_features,
    torch.tensor(pca_E.components_),
    torch.tensor(pca_E.mean_),
    data_files,
)

train_E_samples = torch.tensor(train_E_samples, dtype=torch.float32)
test_E_samples = torch.tensor(test_E_samples, dtype=torch.float32)

train_E_recon = E_encoder.decoder(E_encoder(train_E_samples)).detach()
test_E_recon = E_encoder.decoder(E_encoder(test_E_samples)).detach()
print(train_E_recon.shape, train_E_samples.shape)

rel_error_train = loss_function.L2RelativeError(
    train_E_recon.unsqueeze(-1), train_E_samples.unsqueeze(-1)
).numpy()
rel_error_test = loss_function.L2RelativeError(
    test_E_recon.unsqueeze(-1), test_E_samples.unsqueeze(-1)
).numpy()

print(f"PCA E Train Relative Error: {rel_error_train}")
print(f"PCA E Test Relative Error: {rel_error_test}")

nu_Encoder = PCAEncoder(
    501,
    n_features,
    torch.tensor(pca_nu.components_),
    torch.tensor(pca_nu.mean_),
    data_files,
)

train_nu_samples = torch.tensor(train_nu_samples, dtype=torch.float32)
test_nu_samples = torch.tensor(test_nu_samples, dtype=torch.float32)
train_nu_recon = nu_Encoder.decoder(nu_Encoder(train_nu_samples)).detach()
test_nu_recon = nu_Encoder.decoder(nu_Encoder(test_nu_samples)).detach()
rel_error_train = loss_function.L2RelativeError(
    train_nu_recon.unsqueeze(-1), train_nu_samples.unsqueeze(-1)
).numpy()
rel_error_test = loss_function.L2RelativeError(
    test_nu_recon.unsqueeze(-1), test_nu_samples.unsqueeze(-1)
).numpy()
print(f"PCA nu Train Relative Error: {rel_error_train}")
print(f"PCA nu Test Relative Error: {rel_error_test}")

if not os.path.exists(folder):
    os.makedirs(folder)
torch.save(E_encoder, os.path.join(folder, "ae_E_encoder.pth"))
torch.save(nu_Encoder, os.path.join(folder, "ae_nu_encoder.pth"))
