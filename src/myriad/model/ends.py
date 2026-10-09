"""The parts of the model that stay on the client: token embedding and output head.

Written out rather than borrowed from Transformers so they can be loaded
without building the whole `Gemma4TextModel`. The equivalence tests compare
them against Transformers.
"""

import torch
import torch.nn.functional as F
from torch import nn

from myriad.model.checkpoint import Checkpoint


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Same arithmetic as Gemma4RMSNorm: normalise in float32, scale, cast back.
    normed = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (normed * weight.float()).type_as(x)


class Embedder(nn.Module):
    """Token ids → initial hidden states, plus per-layer inputs on models that have them (E2B/E4B)."""

    def __init__(self, config, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.config = config
        self.embed = weights["embed_tokens.weight"]
        self.ple_dim = config.hidden_size_per_layer_input or 0
        if self.ple_dim:
            self.ple_table = weights["embed_tokens_per_layer.weight"]
            self.ple_projection = weights["per_layer_model_projection.weight"]
            self.ple_norm = weights["per_layer_projection_norm.weight"]

    @staticmethod
    def weight_names(config) -> list[str]:
        names = ["embed_tokens.weight"]
        if config.hidden_size_per_layer_input:
            names += [
                "embed_tokens_per_layer.weight",
                "per_layer_model_projection.weight",
                "per_layer_projection_norm.weight",
            ]
        return names

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint, device="cpu", dtype=torch.bfloat16, ple_device=None) -> "Embedder":
        """`ple_device` can keep the large per-layer-embedding table (4-5 GiB on E2B/E4B) in CPU RAM
        while the rest runs on the GPU; only a few rows of it are read per token."""
        config = checkpoint.text_config()
        names = cls.weight_names(config)
        weights = checkpoint.load([n for n in names if n != "embed_tokens_per_layer.weight"], device, dtype)
        if "embed_tokens_per_layer.weight" in names:
            weights |= checkpoint.load(["embed_tokens_per_layer.weight"], ple_device or device, dtype)
        return cls(config, weights)

    @property
    def device(self) -> torch.device:
        return self.embed.device

    @torch.inference_mode()
    def token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """The scaled token embedding alone ``[1, n, hidden]`` (no per-layer inputs)."""
        # The scale is rounded to the weight dtype first, as in Gemma4TextScaledWordEmbedding.
        scale = torch.tensor(self.config.hidden_size**0.5, dtype=self.embed.dtype)
        return F.embedding(token_ids.to(self.device), self.embed) * scale

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """token_ids ``[1, n]`` → hidden ``[1, n, hidden]`` and per-layer inputs ``[1, n, layers, ple_dim]`` or None."""
        token_ids = token_ids.to(self.device)
        dtype = self.embed.dtype
        hidden = self.token_embedding(token_ids)
        if not self.ple_dim:
            return hidden, None

        n_layers = self.config.num_hidden_layers
        token_part = F.embedding(token_ids.to(self.ple_table.device), self.ple_table).to(self.device)
        token_part = token_part * torch.tensor(self.ple_dim**0.5, dtype=dtype)
        token_part = token_part.reshape(*token_ids.shape, n_layers, self.ple_dim)
        context_part = F.linear(hidden, self.ple_projection) * self.config.hidden_size**-0.5
        context_part = context_part.reshape(*token_ids.shape, n_layers, self.ple_dim)
        context_part = _rms_norm(context_part, self.ple_norm, self.config.rms_norm_eps)
        return hidden, (context_part + token_part) * 2.0**-0.5


class Head(nn.Module):
    """Last hidden states → logits: final norm, output head (tied to the embedding), soft cap."""

    def __init__(self, config, weights: dict[str, torch.Tensor]):
        super().__init__()
        self.config = config
        self.norm = weights["norm.weight"]
        self.output = weights["embed_tokens.weight"]

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint, device="cpu", dtype=torch.bfloat16, embedder: Embedder | None = None) -> "Head":
        """Load the head; pass the client's `embedder` to share the tied weight instead of loading it twice."""
        config = checkpoint.text_config()
        weights = checkpoint.load(["norm.weight"], device, dtype)
        if embedder is not None and embedder.embed.device == torch.device(device):
            weights["embed_tokens.weight"] = embedder.embed
        else:
            weights |= checkpoint.load(["embed_tokens.weight"], device, dtype)
        return cls(config, weights)

    @property
    def device(self) -> torch.device:
        return self.output.device

    @torch.inference_mode()
    def normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        """The final norm: last-layer output → the model's final hidden state (what an MTP drafter reads)."""
        return _rms_norm(hidden.to(self.device), self.norm, self.config.rms_norm_eps)

    @torch.inference_mode()
    def project(self, normed: torch.Tensor) -> torch.Tensor:
        """Final hidden state ``[1, n, hidden]`` → logits ``[1, n, vocab]`` in the model dtype."""
        logits = F.linear(normed, self.output)
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.project(self.normalize(hidden))
