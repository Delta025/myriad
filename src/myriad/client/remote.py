"""`RemotePipeline`: the client side of a swarm.

The client runs the embedding, its own first and last layers, and the output
head. The layers in between run on peers, which the client calls one after
another ("star" topology): client → peer 1 → client → peer 2 → … → client.
Every call is timed, and a `forward` event per call goes to the tracker for
the dashboard.
"""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx
import torch
from websockets.sync.client import connect

from myriad.model.checkpoint import Checkpoint
from myriad.model.ends import Embedder, Head
from myriad.model.kv import KVCache
from myriad.model.pipeline import as_ids
from myriad.model.split import validate_split
from myriad.model.stage import Stage
from myriad.protocol.messages import PROTOCOL_VERSION, decode, encode

log = logging.getLogger("myriad.client")

MAX_FRAME_BYTES = 256 * 2**20
REPLY_TIMEOUT_SECONDS = 120.0


class PeerError(RuntimeError):
    pass


class PeerLink:
    """One WebSocket connection to one peer, used for request/reply calls."""

    def __init__(self, url: str, client_id: str):
        self.url = url
        self._connection = connect(url, max_size=MAX_FRAME_BYTES, compression=None, open_timeout=30)
        self.ws = self._connection.__enter__()  # websockets 17 wants the context-manager protocol
        self._next_id = 0
        info = self.call({"type": "hello", "client_id": client_id})
        if info["protocol"] != PROTOCOL_VERSION:
            raise PeerError(f"{url} speaks protocol {info['protocol']}, expected {PROTOCOL_VERSION}")
        self.peer_id, self.model, self.region = info["peer_id"], info["model"], info["region"]
        self.start, self.end = info["start"], info["end"]

    def call(self, request: dict) -> dict:
        self._next_id += 1
        request = request | {"id": self._next_id}
        self.ws.send(encode(request))
        reply = decode(self.ws.recv(timeout=REPLY_TIMEOUT_SECONDS))
        if reply.get("id") != self._next_id:
            raise PeerError(f"{self.url}: reply id {reply.get('id')} does not match request {self._next_id}")
        if reply["type"] == "error":
            raise PeerError(f"{self.url}: {reply['code']}: {reply['message']}")
        return reply

    def close(self) -> None:
        self._connection.__exit__(None, None, None)


class EventSink:
    """Sends events to the tracker from a background thread, so reporting never slows generation."""

    def __init__(self, tracker_url: str | None):
        self.tracker_url = tracker_url
        self.history: list[dict] = []
        self._queue: queue.Queue = queue.Queue()
        if tracker_url:
            threading.Thread(target=self._send_loop, daemon=True, name="events").start()

    def emit(self, event: dict) -> None:
        self.history.append(event)
        if self.tracker_url:
            self._queue.put(event)

    def _send_loop(self) -> None:
        with httpx.Client(base_url=self.tracker_url, timeout=5.0) as http:
            while True:
                batch = [self._queue.get()]
                while not self._queue.empty() and len(batch) < 100:
                    batch.append(self._queue.get_nowait())
                try:
                    http.post("/events", json=batch)
                except httpx.HTTPError as exc:
                    log.warning("could not send %d events to the tracker: %s", len(batch), exc)


@dataclass
class _LocalPart:
    stage: Stage
    cache: KVCache = field(default_factory=KVCache)


class RemotePipeline:
    """Same interface as `LocalPipeline` (see `myriad.model.pipeline.Pipeline`), with the middle layers on peers."""

    def __init__(
        self,
        embedder: Embedder,
        head: Head,
        first: Stage | None,
        last: Stage | None,
        peers: list[PeerLink],
        events: EventSink | None = None,
        client_id: str | None = None,
    ):
        self.embedder, self.head = embedder, head
        self.first = _LocalPart(first) if first is not None else None
        self.last = _LocalPart(last) if last is not None else None
        self.peers = peers
        self.events = events or EventSink(None)
        self.client_id = client_id or uuid.uuid4().hex[:8]
        self.session = uuid.uuid4().hex
        self.length = 0
        self.last_hidden: torch.Tensor | None = None
        self.last_start_pos = 0

        config = embedder.config
        split = [(p.start, p.end) for p in peers]
        if first is not None:
            split.insert(0, (first.start, first.end))
        if last is not None:
            split.append((last.start, last.end))
        validate_split(config, split)

        for peer in peers:
            peer.call({"type": "open_session", "session": self.session, "client_id": self.client_id})

    @staticmethod
    def load_client_parts(
        checkpoint: Checkpoint, first_layers: int = 1, last_layers: int = 0, dtype: torch.dtype = torch.bfloat16,
        stage_device: str = "cpu", ends_device: str = "cpu",
    ) -> dict:
        """Load what the client runs itself: embedding, head, and its first and last layers."""
        n_layers = checkpoint.text_config().num_hidden_layers
        start, end = first_layers, n_layers - last_layers
        embedder = Embedder.from_checkpoint(checkpoint, ends_device, dtype)
        return dict(
            embedder=embedder,
            head=Head.from_checkpoint(checkpoint, ends_device, dtype, embedder=embedder),
            first=Stage.from_checkpoint(checkpoint, 0, start, stage_device, dtype) if start > 0 else None,
            last=Stage.from_checkpoint(checkpoint, end, n_layers, stage_device, dtype) if end < n_layers else None,
        )

    @classmethod
    def connect(
        cls,
        checkpoint: Checkpoint | str,
        tracker_url: str,
        model: str | None = None,
        first_layers: int = 1,
        last_layers: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        stage_device: str = "cpu",
        ends_device: str = "cpu",
        parts: dict | None = None,
    ) -> "RemotePipeline":
        """Load the client's parts of `checkpoint` and ask the tracker for peers covering the rest.

        `model` is the name peers registered under (default: the checkpoint argument).
        Keeping last layers on E2B/E4B means keeping their whole KV-sharing block.
        `parts` (from `load_client_parts`) reuses already-loaded client weights for a new session.
        """
        if not isinstance(checkpoint, Checkpoint):
            model = model or str(checkpoint)
            checkpoint = Checkpoint(checkpoint)
        elif model is None:
            raise ValueError("pass `model` (the name peers registered under) along with a Checkpoint object")
        n_layers = checkpoint.text_config().num_hidden_layers
        start, end = first_layers, n_layers - last_layers

        route = httpx.get(f"{tracker_url}/route", params={"model": model, "start": start, "end": end}, timeout=10.0)
        if route.status_code == 404:
            raise PeerError(route.json()["detail"])
        route.raise_for_status()

        client_id = uuid.uuid4().hex[:8]
        peers = [PeerLink(hop["url"], client_id) for hop in route.json()]
        parts = parts or cls.load_client_parts(checkpoint, first_layers, last_layers, dtype, stage_device, ends_device)
        return cls(parts["embedder"], parts["head"], parts["first"], parts["last"], peers, EventSink(tracker_url), client_id)

    def forward(self, token_ids, start_pos: int) -> torch.Tensor:
        if start_pos > self.length:
            raise ValueError(f"start_pos {start_pos} is past the {self.length} positions processed")
        t_begin = time.perf_counter()
        ids = as_ids(token_ids)
        hidden, per_layer_inputs = self.embedder(ids)

        def stage_inputs(start, end):
            return per_layer_inputs[:, :, start:end] if per_layer_inputs is not None else None

        if self.first is not None:
            stage = self.first.stage
            hidden = stage(hidden, start_pos, self.first.cache, stage_inputs(stage.start, stage.end))

        hops = []
        for peer in self.peers:
            sent = time.perf_counter()
            request = {"type": "forward", "session": self.session, "start_pos": start_pos, "hidden": hidden.cpu()}
            if per_layer_inputs is not None:
                request["per_layer_inputs"] = stage_inputs(peer.start, peer.end).cpu()
            reply = peer.call(request)
            hidden = reply["hidden"]
            hops.append(
                {
                    "peer_id": peer.peer_id,
                    "region": peer.region,
                    "layers": [peer.start, peer.end],
                    "rtt_ms": round((time.perf_counter() - sent) * 1000, 2),
                    "queue_ms": round(reply["queue_ms"], 2),
                    "compute_ms": round(reply["compute_ms"], 2),
                }
            )

        if self.last is not None:
            stage = self.last.stage
            hidden = stage(hidden, start_pos, self.last.cache, stage_inputs(stage.start, stage.end))
        normed = self.head.normalize(hidden)
        self.last_hidden, self.last_start_pos = normed[0], start_pos
        logits = self.head.project(normed)[0].float().cpu()
        self.length = start_pos + ids.shape[1]

        self.events.emit(
            {
                "type": "forward",
                "time": time.time(),
                "client_id": self.client_id,
                "session": self.session,
                "start_pos": start_pos,
                "n_tokens": ids.shape[1],
                "hops": hops,
                "total_ms": round((time.perf_counter() - t_begin) * 1000, 2),
            }
        )
        return logits

    def truncate(self, length: int) -> None:
        for part in (self.first, self.last):
            if part is not None:
                part.cache.truncate(length)
        for peer in self.peers:
            peer.call({"type": "truncate", "session": self.session, "length": length})
        self.length = min(self.length, length)

    def cached_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V of a layer the client runs itself (peers' caches stay on the peers)."""
        for part in (self.first, self.last):
            if part is not None and part.stage.start <= layer_idx < part.stage.end:
                return part.cache.layer(layer_idx)
        raise KeyError(f"layer {layer_idx} runs on a peer, not on the client")

    def close(self) -> None:
        for peer in self.peers:
            try:
                peer.call({"type": "close_session", "session": self.session})
            finally:
                peer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
