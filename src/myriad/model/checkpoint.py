"""Read a Gemma 4 checkpoint without loading all of it.

A peer that hosts layers 20-40 of 31B should read about a third of the weights,
not 62 GB. `Checkpoint` maps tensor names to safetensors files and loads only
the tensors that are asked for.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, Gemma4TextConfig

_DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.jinja"]


class Checkpoint:
    """A local directory or Hugging Face repo id holding a Gemma 4 checkpoint."""

    def __init__(self, name_or_path: str | Path):
        path = Path(name_or_path)
        if not path.is_dir():
            path = Path(snapshot_download(str(name_or_path), allow_patterns=_DOWNLOAD_PATTERNS))
        self.path = path

        index = path / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self._files = {name: path / file for name, file in weight_map.items()}
        else:
            file = path / "model.safetensors"
            with safe_open(file, "pt") as f:
                self._files = {name: file for name in f.keys()}

        # Released checkpoints are multimodal (`model.language_model.*`); a text-only
        # Gemma4ForCausalLM would use `model.*`.
        self.text_prefix = (
            "model.language_model." if any(n.startswith("model.language_model.") for n in self._files) else "model."
        )

    def text_config(self) -> Gemma4TextConfig:
        config = AutoConfig.from_pretrained(self.path)
        config = config.get_text_config()
        # Stages pass explicit boolean masks, which only the SDPA path accepts.
        config._attn_implementation = "sdpa"
        return config

    def names(self, prefix: str = "") -> list[str]:
        """Text-model tensor names (without the text prefix) that start with `prefix`."""
        full = self.text_prefix + prefix
        return [n[len(self.text_prefix) :] for n in self._files if n.startswith(full)]

    def load(
        self, names: Iterable[str], device: torch.device | str = "cpu", dtype: torch.dtype | None = None
    ) -> dict[str, torch.Tensor]:
        """Load text-model tensors by name (without the text prefix), opening each file once."""
        by_file: dict[Path, list[str]] = {}
        for name in names:
            by_file.setdefault(self._files[self.text_prefix + name], []).append(name)

        tensors = {}
        for file, file_names in by_file.items():
            with safe_open(file, "pt") as f:
                for name in file_names:
                    t = f.get_tensor(self.text_prefix + name)
                    if dtype is not None and t.is_floating_point():
                        t = t.to(dtype)
                    tensors[name] = t.to(device)
        return tensors
