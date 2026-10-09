"""Tiny random-weight Gemma 4 checkpoints for tests.

The checkpoints use the same layout as the real Gemma 4 releases
(`Gemma4ForConditionalGeneration`, weights under `model.language_model.`),
so the loading code under test is the code that loads E4B or 31B.

Two variants cover the architecture features that matter for splitting:

- ``dense``: like 31B. Global layers reuse keys as values (``attention_k_eq_v``)
  and have their own KV head count; no per-layer embeddings, no KV sharing.
- ``ple``: like E2B/E4B. Per-layer embeddings, the last layers share KV with
  earlier ones, and those shared layers have a double-wide MLP.

Both use a sliding window of 4 so short test sequences cross it.
"""

from pathlib import Path

import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

TINY_VOCAB = 512

_COMMON = dict(
    vocab_size=TINY_VOCAB,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    global_head_dim=32,
    sliding_window=4,
    max_position_embeddings=1024,
    final_logit_softcapping=30.0,
    # s s F s s F s F: several global layers, and the last layer is global as in every Gemma 4
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2
    + ["sliding_attention", "full_attention"],
)

VARIANTS = {
    "dense": dict(
        _COMMON,
        attention_k_eq_v=True,
        num_global_key_value_heads=1,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        num_kv_shared_layers=0,
    ),
    "ple": dict(
        _COMMON,
        attention_k_eq_v=False,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=TINY_VOCAB,
        num_kv_shared_layers=3,
        use_double_wide_mlp=True,
    ),
}


def tiny_config(variant: str) -> Gemma4Config:
    return Gemma4Config(text_config=Gemma4TextConfig(**VARIANTS[variant]), vision_config=None, audio_config=None)


@torch.no_grad()
def make_tiny_checkpoint(path: Path, variant: str, seed: int = 0) -> Path:
    """Build a tiny Gemma 4 with random weights and save it to `path` as safetensors."""
    torch.manual_seed(seed)
    model = Gemma4ForConditionalGeneration(tiny_config(variant)).eval()

    # The default init (std 0.02, norms at 1) gives near-uniform logits, where greedy
    # decoding is decided by noise. Spread the weights so tokens are clearly ranked.
    for name, param in model.named_parameters():
        if name.endswith("norm.weight"):
            param.uniform_(0.5, 1.5)
        else:
            param.normal_(0.0, 0.15)
    for name, buf in model.named_buffers():
        if name.endswith("layer_scalar"):
            buf.uniform_(0.5, 1.5)  # make sure the loader picks it up from the checkpoint

    model.save_pretrained(path)
    return path


class ThreadedSwarm:
    """A tracker and peers on localhost, served from one background event loop, for tests.

    Real deployments run each peer as its own process (`myriad peer`); the
    protocol and code paths are the same.
    """

    def __init__(self, checkpoint, split, model: str, dtype=torch.float32, delays_ms=None):
        import asyncio
        import socket
        import threading
        import time

        import httpx
        import uvicorn

        from myriad.model.checkpoint import Checkpoint
        from myriad.model.stage import Stage
        from myriad.peer.server import PeerServer
        from myriad.tracker.app import create_app

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="swarm").start()

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.tracker_url = f"http://127.0.0.1:{port}"
        self._uvicorn = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
        self._tasks = [asyncio.run_coroutine_threadsafe(self._uvicorn.serve(), self.loop)]
        while not self._uvicorn.started:
            time.sleep(0.01)

        ckpt = Checkpoint(checkpoint)
        delays_ms = delays_ms or [0.0] * len(split)
        self.peers = [
            PeerServer(Stage.from_checkpoint(ckpt, a, b, "cpu", dtype), model, region=f"r{i}", delay_ms=d)
            for i, ((a, b), d) in enumerate(zip(split, delays_ms))
        ]
        for peer in self.peers:
            self._tasks.append(asyncio.run_coroutine_threadsafe(peer.serve(tracker_url=self.tracker_url), self.loop))

        deadline = time.monotonic() + 30
        while len(httpx.get(f"{self.tracker_url}/peers").json()) < len(self.peers):
            if time.monotonic() > deadline:
                raise TimeoutError("peers did not register with the tracker")
            time.sleep(0.05)

    def close(self) -> None:
        import concurrent.futures

        self._uvicorn.should_exit = True
        for task in self._tasks[1:]:
            task.cancel()
        # Let the tasks finish their cleanup before stopping the loop.
        concurrent.futures.wait(self._tasks, timeout=10)
        self.loop.call_soon_threadsafe(self.loop.stop)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
