"""A simple test for HyperICNN / LoRAICNN on learning a rotated quadratic function.

Run:
	python test.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from loraicnn import LoRAICNN
from loraicnn_utils import (
	ICNNArchitecture,
	MLPAdapter,
	lora_adapter_param_dim,
)

def ground_truth(y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
	"""The rotating anisotropic quadratic bowl.

	Args:
		y: Tensor [B,2]
		theta: Tensor [B,1]

	Returns:
		Tensor [B,1]
	"""
	cos_theta = torch.cos(theta)
	sin_theta = torch.sin(theta)
	# IMPORTANT: keep y columns as [B,1] to avoid broadcasting to [B,B].
	y_rot_1 = y[:, 0:1] * cos_theta + y[:, 1:2] * sin_theta
	y_rot_2 = -y[:, 0:1] * sin_theta + y[:, 1:2] * cos_theta
	return 0.5 * y_rot_1**2 + 0.05 * y_rot_2**2

def train_model(
	model: nn.Module,
	*,
	name: str,
	steps: int = 50000,
	lr: float = 1e-4,
	batch_size: int = 128,
	device: str = "cpu",
) -> nn.Module:
	model.to(device)
	model.train()
	optimizer = optim.AdamW(model.parameters(), lr=lr)

	for i in range(steps):
		y = torch.randn(batch_size, 2, device=device)*4.0
		theta = torch.rand(batch_size, 1, device=device) * torch.pi

		target = ground_truth(y, theta)
		# Our models take x plus a keyword-only conditioning payload.
		pred = model(y, cond=theta)

		loss = torch.mean((pred - target) ** 2 / (target**2 + 1e-2))

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		if i % 250 == 0:
			print(f"{name} Step {i}, Loss: {loss.item():.6f}")

	print(f"{name} Final Loss: {loss.item():.6f}")
	return model

@torch.no_grad()
def eval_samples(model: nn.Module, *, device: str = "cpu") -> None:
	model.eval()
	y = torch.randn(256, 2, device=device)
	u = torch.rand(256, 1, device=device) * torch.pi
	target = ground_truth(y, u)
	pred = model(y, cond=u)

	# Relative L2 over samples
	num = torch.sum((pred - target) ** 2)
	den = torch.sum(target**2) + 1e-12
	rel = torch.sqrt(num / den)
	print(f"Relative L2: {float(rel):.6f}")


@torch.no_grad()
def relative_l2_error_over_grid_and_angles(
	*, model: nn.Module, device: str, n_grid: int = 50, angles_deg: list[int] | None = None
) -> float:
	"""Match picnn_vs_hypernet's evaluation (grid + angles, mean over angles)."""
	model.eval()
	if angles_deg is None:
		angles_deg = list(range(0, 181, 10))
	angles = torch.tensor([a * torch.pi / 180.0 for a in angles_deg], device=device)

	# Grid in y
	lin = torch.linspace(-4.0, 4.0, n_grid, device=device)
	Y1, Y2 = torch.meshgrid(lin, lin, indexing="ij")
	grid_y = torch.stack([Y1.reshape(-1), Y2.reshape(-1)], dim=1)  # [N,2]

	errs: list[float] = []
	for theta in angles:
		u = torch.full((grid_y.shape[0], 1), float(theta), device=device)
		gt = ground_truth(grid_y, u)
		pr = model(grid_y, cond=u)
		diff = pr - gt
		num = float(torch.sum(diff * diff).detach().cpu())
		den = float(torch.sum(gt * gt).detach().cpu())
		if den <= 0.0:
			errs.append(float("inf"))
		else:
			errs.append((num / den) ** 0.5)
	return float(sum(errs) / len(errs))


def count_trainable_params(model: nn.Module) -> int:
	return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main() -> None:
	torch.manual_seed(0)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")

	arch = ICNNArchitecture(input_dim=2, hidden_dims=(4, 4, 4), output_dim=1)

	print("\nTraining LoRAICNN...")
	print(f"LoRAICNN adapter output dim: {lora_adapter_param_dim(arch, rank=2)}")
	adapter = MLPAdapter(cond_dim=1, hidden=32)
	lora = LoRAICNN(arch, adapter, rank=4, alpha=1.0, activation="softplus")
	print(f"LoRAICNN trainable params: {count_trainable_params(lora)}")
	train_model(lora, name="LoRAICNN", device=device)
	eval_samples(lora, device=device)
	print(
		f"LoRAICNN relative error: {relative_l2_error_over_grid_and_angles(model=lora, device=device):.6f}"
	)


if __name__ == "__main__":
	main()

