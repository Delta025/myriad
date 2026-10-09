"""Download single tensors from a safetensors file on the Hugging Face Hub, by byte range.

A safetensors file is an 8-byte header length, a JSON header giving every
tensor's dtype, shape and byte offsets, then the raw data. So a peer that needs
half the layers of a model can read the header and fetch just those bytes; it
need not download whole shards (Gemma 4 31B comes as two 29 GiB files, and every
peer would otherwise need both).

Fetched tensors are cached on disk, one raw file each.
"""

import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import torch
from huggingface_hub import constants

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
          "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool}
CHUNK = 64 * 2**20

RangeReader = Callable[[str, int, int], bytes]  # (file name, start, end exclusive) -> bytes


def hub_reader(repo_id: str, revision: str) -> RangeReader:
    """Read byte ranges of a file in a Hub repo (public, or with the token from the environment)."""
    headers = {}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    client = httpx.Client(follow_redirects=True, timeout=120.0, headers=headers)

    def read(file: str, start: int, end: int) -> bytes:
        url = f"{constants.ENDPOINT}/{repo_id}/resolve/{revision}/{file}"
        response = client.get(url, headers={"Range": f"bytes={start}-{end - 1}"})
        response.raise_for_status()
        if len(response.content) != end - start:
            raise OSError(f"short read of {file} [{start}, {end}): got {len(response.content)} bytes")
        return response.content

    return read


class RemoteTensors:
    def __init__(self, read: RangeReader, cache_dir: Path, workers: int = 8):
        self.read, self.cache_dir, self.workers = read, cache_dir, workers
        self._headers: dict[str, tuple[dict, int]] = {}

    def header(self, file: str) -> tuple[dict, int]:
        """The file's tensor table and the offset where tensor data starts."""
        if file not in self._headers:
            cached = self.cache_dir / f"{file}.header.json"
            if cached.exists():
                table = json.loads(cached.read_text())
            else:
                size = int.from_bytes(self.read(file, 0, 8), "little")
                table = json.loads(self.read(file, 8, 8 + size))
                table.pop("__metadata__", None)
                table["__data_start__"] = 8 + size
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(table))
            self._headers[file] = (table, table["__data_start__"])
        return self._headers[file]

    def _path(self, name: str) -> Path:
        return self.cache_dir / "tensors" / f"{name}.bin"

    def fetch(self, file: str, names: Iterable[str]) -> None:
        """Download the named tensors of `file` that are not cached yet, in parallel chunks."""
        table, data_start = self.header(file)
        missing = [n for n in names if not self._path(n).exists()]
        jobs = []  # (partial file, offset within the tensor, absolute start, absolute end)
        for name in missing:
            begin, end = table[name]["data_offsets"]
            partial = self._path(name).with_suffix(".part")
            partial.parent.mkdir(parents=True, exist_ok=True)
            with open(partial, "wb") as out:
                out.truncate(end - begin)
            jobs += [(partial, o, data_start + begin + o, data_start + min(begin + o + CHUNK, end))
                     for o in range(0, end - begin, CHUNK)]

        def download(job) -> None:
            partial, offset, start, end = job
            data = self.read(file, start, end)
            with open(partial, "r+b") as out:
                out.seek(offset)
                out.write(data)

        with ThreadPoolExecutor(self.workers) as pool:
            list(pool.map(download, jobs))
        for name in missing:
            self._path(name).with_suffix(".part").replace(self._path(name))  # only complete files count as cached

    def load(self, file: str, name: str) -> torch.Tensor:
        table, _ = self.header(file)
        info = table[name]
        dtype, shape = DTYPES[info["dtype"]], info["shape"]
        data = self._path(name).read_bytes()
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape) if data else torch.empty(shape, dtype=dtype)
