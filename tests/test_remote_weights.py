"""Fetching single tensors by byte range gives exactly the tensors in the file."""

import torch
from safetensors import safe_open

from myriad.model import remote_weights
from myriad.model.remote_weights import RemoteTensors


def local_reader(directory, log):
    def read(file, start, end):
        log.append((file, start, end))
        with open(directory / file, "rb") as f:
            f.seek(start)
            return f.read(end - start)
    return read


def test_range_fetch_matches_file(tiny_checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(remote_weights, "CHUNK", 1000)  # force multi-chunk downloads on tiny tensors
    log = []
    remote = RemoteTensors(local_reader(tiny_checkpoint, log), tmp_path / "cache")
    wanted = ["model.language_model.layers.3.mlp.up_proj.weight", "model.language_model.layers.3.layer_scalar"]
    remote.fetch("model.safetensors", wanted)
    with safe_open(tiny_checkpoint / "model.safetensors", "pt") as f:
        for name in wanted:
            assert torch.equal(remote.load("model.safetensors", name), f.get_tensor(name))
    fetched = sum(end - start for _, start, end in log)
    total = (tiny_checkpoint / "model.safetensors").stat().st_size
    assert fetched < total / 10  # only the header and those two tensors were read

    log.clear()
    remote.fetch("model.safetensors", wanted)  # cached now
    assert log == []
