"""Read a Gemma 4 checkpoint without downloading or loading all of it.

A peer that hosts layers 20-40 of 31B should fetch and read about a third of the
weights, not 62 GB. `Checkpoint` maps tensor names to safetensors files and loads
only the tensors that are asked for. For a model on the Hub it downloads only the
metadata up front; weights come later, tensor by tensor, by HTTP range requests
(see `remote_weights.py`), unless a whole shard is already on disk.
"""

import json
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, Gemma4TextConfig

from myriad.model.remote_weights import RemoteTensors, hub_reader

_METADATA_PATTERNS = ["*.json", "tokenizer*", "*.jinja"]
CACHE = Path(os.environ.get("MYRIAD_CACHE", Path.home() / ".cache" / "myriad"))


class Checkpoint:
    """A local directory or Hugging Face repo id holding a Gemma 4 checkpoint."""

    def __init__(self, name_or_path: str | Path):
        path = Path(name_or_path)
        self.repo_id = None
        if not path.is_dir():
            # Metadata only (config, index, tokenizer); weight shards come later, on demand.
            self.repo_id = str(name_or_path)
            path = Path(snapshot_download(self.repo_id, allow_patterns=_METADATA_PATTERNS))
        self.path = path
        self._remote = None
        if self.repo_id is not None:
            revision = path.name  # the snapshot directory is named after the commit
            cache = CACHE / self.repo_id.replace("/", "--") / revision
            self._remote = RemoteTensors(hub_reader(self.repo_id, revision), cache)

        index = path / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self._files = {name: path / file for name, file in weight_map.items()}
        elif (path / "model.safetensors").exists():
            with safe_open(path / "model.safetensors", "pt") as f:
                self._files = {name: path / "model.safetensors" for name in f.keys()}
        else:
            table, _ = self._remote.header("model.safetensors")
            self._files = {name: path / "model.safetensors" for name in table if not name.startswith("__")}

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

    def _local(self, file: Path) -> Path:
        """Make sure a weight file is on disk, downloading it from the Hub if needed."""
        if not file.exists() and self.repo_id is not None:
            # The snapshot directory is named after the commit, so this fetches the same revision.
            hf_hub_download(self.repo_id, file.name, revision=self.path.name)
        return file

    def download_all(self) -> None:
        """Fetch every weight file, for code that loads the whole model (such as the Transformers reference)."""
        for file in sorted(set(self._files.values())):
            self._local(file)

    def rows(self, name: str) -> "LazyRows":
        """A large 2-D tensor whose rows are read from disk only when asked for."""
        return LazyRows(self._local(self._files[self.text_prefix + name]), self.text_prefix + name)

    def load(
        self, names: Iterable[str], device: torch.device | str = "cpu", dtype: torch.dtype | None = None
    ) -> dict[str, torch.Tensor]:
        """Load text-model tensors by name (without the text prefix), opening each file once."""
        by_file: dict[Path, list[str]] = {}
        for name in names:
            by_file.setdefault(self._files[self.text_prefix + name], []).append(name)

        tensors = {}
        for file, file_names in by_file.items():
            full_names = [self.text_prefix + n for n in file_names]
            if file.exists() or self._remote is None:
                with safe_open(file, "pt") as f:
                    raw = {n: f.get_tensor(full) for n, full in zip(file_names, full_names)}
            else:
                self._remote.fetch(file.name, full_names)
                raw = {n: self._remote.load(file.name, full) for n, full in zip(file_names, full_names)}
            for name, t in raw.items():
                if dtype is not None and t.is_floating_point():
                    t = t.to(dtype)
                tensors[name] = t.to(device)
        return tensors


class LazyRows:
    """Rows of a tensor in a safetensors file, read on demand with plain file reads.

    E2B/E4B's per-layer-embedding table is 4-5 GiB, yet a token needs only its own
    row. Reading rows from the (OS-cached) file keeps it out of the client's RAM.
    It deliberately avoids memory-mapping: on Windows a mapping of the whole file
    counts against the commit limit for as long as it stays open.
    """

    _DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

    def __init__(self, file: Path, name: str):
        self._file = open(file, "rb")
        header_size = int.from_bytes(self._file.read(8), "little")
        info = json.loads(self._file.read(header_size))[name]
        self.dtype = self._DTYPES[info["dtype"]]
        self.shape = tuple(info["shape"])
        self._row_bytes = self.shape[1] * torch.tensor([], dtype=self.dtype).element_size()
        self._base = 8 + header_size + info["data_offsets"][0]

    def row(self, index: int) -> torch.Tensor:
        self._file.seek(self._base + index * self._row_bytes)
        return torch.frombuffer(bytearray(self._file.read(self._row_bytes)), dtype=self.dtype)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Rows for ``token_ids`` of any shape → ``[*token_ids.shape, row_size]``."""
        flat = token_ids.reshape(-1).tolist()
        unique = sorted(set(flat))
        rows = torch.stack([self.row(i) for i in unique])
        index = torch.tensor([unique.index(i) for i in flat])
        return rows[index].reshape(*token_ids.shape, -1)
