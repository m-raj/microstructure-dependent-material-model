
"""Utilities for LoRA-ICNN models.

This module intentionally contains everything that is *not* the main model
class, so your training loop typically imports `LoRAICNN` from `loraicnn.py`
and uses helpers from here only when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


Activation = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ICNNArchitecture:
	"""Defines the ICNN layer sizes.

	Attributes:
		input_dim: Dimension of input x.
		hidden_dims: Hidden z dimensions (must contain at least one layer).
		output_dim: Dimension of output y (often 1).
	"""

	input_dim: int
	hidden_dims: tuple[int, ...]
	output_dim: int = 1

	def __post_init__(self) -> None:
		"""Validate that the architecture is well-formed.

		Raises:
			ValueError: If any dimension is non-positive, or if `hidden_dims` is empty.
		"""
		if self.input_dim <= 0:
			raise ValueError("input_dim must be > 0")
		if self.output_dim <= 0:
			raise ValueError("output_dim must be > 0")
		if len(self.hidden_dims) < 1:
			raise ValueError("hidden_dims must contain at least one layer")
		if any(h <= 0 for h in self.hidden_dims):
			raise ValueError("hidden_dims must all be > 0")


@dataclass(frozen=True)
class ParamBlock:
	"""A single named parameter block with a fixed shape."""

	name: str
	shape: torch.Size

	@property
	def numel(self) -> int:
		"""Number of scalars in this block (product of its shape)."""
		return int(prod(self.shape))


def _flatten_spec_numel(spec: Iterable[ParamBlock]) -> int:
	"""Total number of scalars across a list of parameter blocks."""
	return int(sum(b.numel for b in spec))


def lora_adapter_param_spec(arch: ICNNArchitecture, *, rank: int) -> list[ParamBlock]:
	"""Spec for flattened outputs of a LoRA adapter.

	For each ICNN weight matrix W (Wx and Wz_raw), the adapter outputs factors:
	  - "<name>_A" of shape [rank, in]
	  - "<name>_B" of shape [out, rank]
	For each bias vector b, the adapter outputs a direct delta:
	  - "<name>_delta" of shape [out]

	The ICNN computes:
	  W_eff = W_base + (B @ A) * (alpha / rank)
	  b_eff = b_base + b_delta

	Note: `.Wz_raw` is treated like a weight matrix; after applying LoRA, the
	ICNN applies `softplus` to enforce nonnegativity.

	Args:
		arch: ICNN layer sizes.
		rank: LoRA rank $r$.

	Returns:
		A list of ParamBlock entries describing the flattened adapter layout.

	Raises:
		ValueError: If rank <= 0.
	"""

	if rank <= 0:
		raise ValueError("rank must be > 0")

	spec: list[ParamBlock] = []
	prev_z: Optional[int] = None
	for i, z_dim in enumerate(arch.hidden_dims):
		# Wx: [z_dim, input]
		spec.append(ParamBlock(f"layers_{i}_Wx_A", torch.Size([rank, arch.input_dim])))
		spec.append(ParamBlock(f"layers_{i}_Wx_B", torch.Size([z_dim, rank])))
		spec.append(ParamBlock(f"layers_{i}_b_delta", torch.Size([z_dim])))
		if prev_z is not None:
			# Wz_raw: [z_dim, prev_z]
			spec.append(ParamBlock(f"layers_{i}_Wz_raw_A", torch.Size([rank, prev_z])))
			spec.append(ParamBlock(f"layers_{i}_Wz_raw_B", torch.Size([z_dim, rank])))
		prev_z = z_dim

	assert prev_z is not None
	# out weights
	spec.append(ParamBlock("out_Wx_A", torch.Size([rank, arch.input_dim])))
	spec.append(ParamBlock("out_Wx_B", torch.Size([arch.output_dim, rank])))
	spec.append(ParamBlock("out_b_delta", torch.Size([arch.output_dim])))
	spec.append(ParamBlock("out_Wz_raw_A", torch.Size([rank, prev_z])))
	spec.append(ParamBlock("out_Wz_raw_B", torch.Size([arch.output_dim, rank])))
	return spec


def lora_adapter_param_dim(arch: ICNNArchitecture, *, rank: int) -> int:
	"""Total flattened parameter dimension for LoRAICNN's adapter output."""
	return _flatten_spec_numel(lora_adapter_param_spec(arch, rank=rank))


def unflatten(spec: list[ParamBlock], flat: torch.Tensor) -> dict[str, torch.Tensor]:
	"""Split a flat vector/tensor into named blocks.

	Supports:
	- flat shape [P] -> returns tensors without batch dim
	- flat shape [B, P] -> returns tensors with leading batch dim [B, ...]

	Why this exists:
		The LoRA adapter outputs a single flat vector; this helper reshapes it into
		the per-layer weight/bias tensors used during the forward pass.

	Args:
		spec: Parameter block spec (names + shapes) defining how to split `flat`.
		flat: Generator/adapter output, shape [P] or [B,P].

	Returns:
		A dict mapping block name -> tensor shaped like the spec.
		If `flat` is batched, values have a leading batch dim.

	Raises:
		ValueError: If `flat` has the wrong rank or wrong total dimension.
	"""

	if flat.ndim == 1:
		flat2 = flat.unsqueeze(0)
		batched = False
	elif flat.ndim == 2:
		flat2 = flat
		batched = True
	else:
		raise ValueError("Expected flat to have shape [P] or [B, P]")

	B, P = flat2.shape
	need = _flatten_spec_numel(spec)
	if P != need:
		raise ValueError(f"Bad generator output dim: expected {need}, got {P}")

	out: dict[str, torch.Tensor] = {}
	offset = 0
	for block in spec:
		n = block.numel
		chunk = flat2[:, offset : offset + n]
		offset += n
		out[block.name] = chunk.view(B, *block.shape)

	if not batched:
		return {k: v[0] for k, v in out.items()}
	return out


def linear_maybe_batched(x: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
	"""Apply a linear map, supporting either shared or per-sample parameters.

	Why this exists:
		In standard PyTorch, `F.linear(x, W, b)` assumes a single weight matrix
		shared across the whole batch (`W` is [out,in]). For some adapters, you may
		produce a different `W_i` per sample, i.e. `W` shaped [B,out,in]. PyTorch
		does not support that in `F.linear`, so we use `torch.bmm` in that case.

	Args:
		x: Input tensor of shape [B, in].
		W: Weight tensor of shape [out, in] or [B, out, in].
		b: Bias tensor of shape [out] or [B, out] or None.

	Returns:
		Tensor of shape [B, out].
	"""

	if W.ndim == 2:
		return F.linear(x, W, b)
	if W.ndim != 3:
		raise ValueError("W must be [out,in] or [B,out,in]")
	if x.ndim != 2:
		raise ValueError("x must be [B,in]")
	B = x.shape[0]
	if W.shape[0] != B:
		raise ValueError("Batch mismatch between x and W")
	# [B,out,in] @ [B,in,1] -> [B,out,1]
	out = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
	if b is None:
		return out
	return out + b


def default_init_linear(weight: torch.Tensor) -> None:
	"""Initialize a weight matrix for the base ICNN.

	Why this exists:
		Keeps initialization consistent and in one place.

	Args:
		weight: Tensor to initialize in-place.
	"""

	nn.init.xavier_uniform_(weight)


def resolve_activation(activation: str | Activation) -> Activation:
	"""Resolve activation to a callable.

	Why this exists:
		To keep the ICNN API minimal, we accept either a real activation callable
		or a short string name for common choices.

	Args:
		activation: Callable or a string shortcut ("softplus" or "relu").

	Returns:
		A callable `act(x)->Tensor`.

	Raises:
		ValueError: If activation is a string but not recognized.
		TypeError: If activation is neither a supported string nor callable.
	"""

	if isinstance(activation, str):
		key = activation.lower()
		if key == "softplus":
			return F.softplus
		if key == "relu":
			return F.relu
		raise ValueError("activation must be a callable or one of: 'softplus', 'relu'")

	if not callable(activation):
		raise TypeError("activation must be callable")
	return activation


class MLPAdapter(nn.Module):
	"""A LoRA adapter that learns its output dim via `configure(spec)`.

	This is a simple example adapter. You can use your own adapter module instead.
	This MLP adapter expects `cond` to be a tensor.

	- forward(cond) -> [B,P]
	- configure(spec) builds the last layer with P = sum(numel(spec))
	"""

	def __init__(self, cond_dim: int, hidden: int = 64):
		"""Create a small MLP adapter.

		Args:
			cond_dim: Dimension of the conditioning vector passed to forward.
			hidden: Hidden width of the MLP.
		"""
		super().__init__()
		self.cond_dim = int(cond_dim)
		self.hidden = int(hidden)
		self.net: Optional[nn.Sequential] = None

	def configure(self, spec: list[ParamBlock]) -> None:
		"""Configure output size based on an ICNN parameter spec.

		Why this exists:
			Your generator/adapter needs to know the required output dimension. The
			ICNN calls this once (if present) so the module can build its last layer.

		Args:
			spec: List of ParamBlock entries; total output dim is sum(block.numel).
		"""

		out_dim = _flatten_spec_numel(spec)
		self.net = nn.Sequential(
			nn.Linear(self.cond_dim, self.hidden),
			nn.GELU(),
			nn.Linear(self.hidden, self.hidden),
			nn.GELU(),
			nn.Linear(self.hidden, self.hidden),
			nn.GELU(),
			nn.Linear(self.hidden, out_dim),
		)

	def forward(self, cond: Optional[torch.Tensor]) -> torch.Tensor:
		"""Generate a flat parameter vector from conditioning input.

		Args:
			cond: Conditioning tensor of shape [cond_dim] or [B, cond_dim].

		Returns:
			Tensor of shape [B, out_dim].
		"""

		if self.net is None:
			raise RuntimeError("MLPAdapter not configured yet (missing configure(spec))")
		if cond is None:
			raise ValueError("cond is required")
		if cond.ndim == 1:
			cond = cond.unsqueeze(0)
		if cond.ndim != 2:
			raise ValueError("cond must be [cond_dim] or [B,cond_dim]")
		return self.net(cond)


__all__ = [
	"Activation",
	"ICNNArchitecture",
	"ParamBlock",
	"default_init_linear",
	"linear_maybe_batched",
	"lora_adapter_param_dim",
	"lora_adapter_param_spec",
	"resolve_activation",
	"unflatten",
	"MLPAdapter",
]

