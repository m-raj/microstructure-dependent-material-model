"""ICNNs with LoRA adapters implemented in PyTorch.

This file provides:

- LoRAICNN: adapter.forward(cond) -> flattened LoRA factors + bias deltas

How the adapter learns the required output size:

- The ICNN computes a small `spec` (a list of parameter blocks with shapes).
- If the provided adapter defines an OPTIONAL method
	`configure(spec)`, the ICNN will call it once in `__init__`.
	This is the only non-forward interaction point.

If you do not want any special method at all, simply construct your adapter
with the correct output dimension ahead of time (you can compute it using
`loraicnn_utils.lora_adapter_param_dim(...)`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loraicnn_utils import (
	Activation,
	ICNNArchitecture,
	default_init_linear,
	linear_maybe_batched,
	lora_adapter_param_spec,
	resolve_activation,
	unflatten,
)

class LoRAICNN(nn.Module):
	"""ICNN with trainable base params plus LoRA deltas from adapter.forward(cond).

	Why `ICNNArchitecture` is required:
	- It fixes all layer sizes (base parameter shapes) and defines the exact LoRA
	  adapter output layout used by `lora_adapter_param_spec(...)` + `unflatten(...)`.

	Adapter contract (minimal):
	- forward(cond) returns a flat tensor of shape: [P] or [B,P]
	  where P = lora_adapter_param_dim(arch, rank)

	Optional one-time hook:
	- configure(spec) to let the adapter build its last layer correctly.
	"""

	def __init__(
		self,
		arch: ICNNArchitecture,
		adapter: nn.Module,
		*,
		rank: int,
		activation: str | Activation = F.softplus,
		alpha: float = 1.0,
	) -> None:
		"""Create an ICNN with trainable base parameters plus LoRA adaptations.

		Args:
			arch: ICNN layer sizes.
			adapter: Module called as `adapter(cond)` that returns a flat tensor of
				shape [P] or [B,P], where P = lora_adapter_param_dim(arch, rank=rank).
			rank: LoRA rank.
			activation: Callable activation.
			alpha: LoRA scaling factor. Effective scaling is alpha / rank.
		"""
		super().__init__()
		if int(rank) <= 0:
			raise ValueError("rank must be > 0")
		self.arch = arch
		self.activation = activation
		self._activation_fn: Activation = resolve_activation(activation)
		self.rank = int(rank)
		self.alpha = float(alpha)
		self.adapter = adapter

		# Base ICNN parameters.
		self.base: nn.ParameterDict = nn.ParameterDict()

		# Hidden layers
		prev_z: int | None = None
		for i, z_dim in enumerate(arch.hidden_dims):
			Wx = nn.Parameter(torch.empty(z_dim, arch.input_dim))
			b = nn.Parameter(torch.zeros(z_dim))
			default_init_linear(Wx)
			self.base[f"layers_{i}_Wx"] = Wx
			self.base[f"layers_{i}_b"] = b

			if prev_z is not None:
				Wz_raw = nn.Parameter(torch.empty(z_dim, prev_z))
				default_init_linear(Wz_raw)
				self.base[f"layers_{i}_Wz_raw"] = Wz_raw

			prev_z = z_dim

		assert prev_z is not None
		out_Wx = nn.Parameter(torch.empty(arch.output_dim, arch.input_dim))
		out_Wz_raw = nn.Parameter(torch.empty(arch.output_dim, prev_z))
		out_b = nn.Parameter(torch.zeros(arch.output_dim))
		default_init_linear(out_Wx)
		default_init_linear(out_Wz_raw)
		self.base["out_Wx"] = out_Wx
		self.base["out_Wz_raw"] = out_Wz_raw
		self.base["out_b"] = out_b

		self._adapter_spec = lora_adapter_param_spec(arch, rank=self.rank)
		configure = getattr(self.adapter, "configure", None)
		if callable(configure):
			configure(self._adapter_spec)

	def _lora_delta(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
		"""Compute a LoRA weight delta from low-rank factors.

		Given factors A and B, computes:
		  DeltaW = (B @ A) * (alpha / rank)

		Args:
			A: Shape [rank, in] or [B, rank, in].
			B: Shape [out, rank] or [B, out, rank].

		Returns:
			DeltaW: Shape [out, in] or [B, out, in].
		"""
		scale = self.alpha / float(self.rank)
		if A.ndim == 2 and B.ndim == 2:
			if A.shape[0] != self.rank or B.shape[1] != self.rank:
				raise ValueError("LoRA rank mismatch between adapter output and model rank")
			return (B @ A) * scale
		if A.ndim == 3 and B.ndim == 3:
			if A.shape[1] != self.rank or B.shape[2] != self.rank:
				raise ValueError("LoRA rank mismatch between adapter output and model rank")
			return torch.bmm(B, A) * scale
		raise ValueError("A/B must be both 2D or both 3D")

	def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None) -> torch.Tensor:
		"""Compute ICNN output for inputs x, using base params + LoRA deltas.

		Args:
			x: Input tensor of shape [B, input_dim].
			cond: Conditioning tensor passed to `adapter(cond)`.

		Returns:
			Tensor of shape [B, output_dim].
		"""
		if x.ndim != 2 or x.shape[-1] != self.arch.input_dim:
			raise ValueError(f"x must be [B,{self.arch.input_dim}]")

		flat = self.adapter(cond)
		flat = flat.to(device=x.device, dtype=x.dtype)
		p = unflatten(self._adapter_spec, flat)

		act = self._activation_fn

		z: torch.Tensor | None = None
		prev_z: int | None = None
		for i, z_dim in enumerate(self.arch.hidden_dims):
			Wx_base = self.base[f"layers_{i}_Wx"]
			b_base = self.base[f"layers_{i}_b"]
			Wx_delta = self._lora_delta(p[f"layers_{i}_Wx_A"], p[f"layers_{i}_Wx_B"])
			b_delta = p[f"layers_{i}_b_delta"]
			Wx_eff = Wx_base + Wx_delta
			b_eff = b_base + b_delta
			pre = linear_maybe_batched(x, Wx_eff, b_eff)

			if prev_z is not None:
				assert z is not None
				Wz_base = self.base[f"layers_{i}_Wz_raw"]
				Wz_delta = self._lora_delta(p[f"layers_{i}_Wz_raw_A"], p[f"layers_{i}_Wz_raw_B"])
				Wz_pos = F.softplus(Wz_base + Wz_delta)
				pre = pre + linear_maybe_batched(z, Wz_pos, None)

			z = act(pre)
			prev_z = z_dim

		assert z is not None
		out_Wx_base = self.base["out_Wx"]
		out_b_base = self.base["out_b"]
		out_Wx_delta = self._lora_delta(p["out_Wx_A"], p["out_Wx_B"])
		out_b_delta = p["out_b_delta"]
		out = linear_maybe_batched(x, out_Wx_base + out_Wx_delta, out_b_base + out_b_delta)

		out_Wz_base = self.base["out_Wz_raw"]
		out_Wz_delta = self._lora_delta(p["out_Wz_raw_A"], p["out_Wz_raw_B"])
		out = out + linear_maybe_batched(z, F.softplus(out_Wz_base + out_Wz_delta), None)
		return out


__all__ = [
	"LoRAICNN",
]

