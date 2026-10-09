"""A peer: hosts one stage (a range of layers) and serves it over WebSockets.

Each client session has its own KV cache. Stage computations run one at a
time on a single worker thread, in arrival order (M5 replaces this with a
queue ordered by credits). The asyncio loop only handles I/O.
"""

import asyncio
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from myriad.model.kv import KVCache
from myriad.model.stage import Stage
from myriad.protocol.messages import PROTOCOL_VERSION, decode, encode, error

log = logging.getLogger("myriad.peer")

HEARTBEAT_SECONDS = 5.0
MAX_FRAME_BYTES = 256 * 2**20


@dataclass
class Session:
    cache: KVCache = field(default_factory=KVCache)
    client_id: str = ""
    last_used: float = field(default_factory=time.monotonic)


class PeerServer:
    def __init__(
        self,
        stage: Stage,
        model: str,
        region: str = "local",
        delay_ms: float = 0.0,
        session_timeout: float = 600.0,
        peer_id: str | None = None,
    ):
        """`delay_ms` is added before every reply, to simulate the round-trip time to a distant peer."""
        self.stage, self.model, self.region = stage, model, region
        self.delay_ms, self.session_timeout = delay_ms, session_timeout
        self.peer_id = peer_id or uuid.uuid4().hex[:8]
        self.sessions: dict[str, Session] = {}
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage")
        self.url: str | None = None  # set by serve()

    # --- request handling ---

    async def handle(self, request: dict) -> dict:
        kind = request["type"]
        reply = {"id": request.get("id")}
        if kind == "hello":
            return reply | {
                "type": "hello_ok",
                "peer_id": self.peer_id,
                "model": self.model,
                "start": self.stage.start,
                "end": self.stage.end,
                "region": self.region,
                "protocol": PROTOCOL_VERSION,
            }
        if kind == "open_session":
            self.sessions[request["session"]] = Session(client_id=request.get("client_id", ""))
            return reply | {"type": "ok"}

        session = self.sessions.get(request.get("session"))
        if session is None:
            return error(request, "unknown_session", f"no session {request.get('session')!r}")
        session.last_used = time.monotonic()

        if kind == "forward":
            received = time.perf_counter()
            hidden, queue_ms, compute_ms = await asyncio.get_running_loop().run_in_executor(
                self._gpu, self._run_stage, session, request, received
            )
            return reply | {"type": "forward_ok", "hidden": hidden, "queue_ms": queue_ms, "compute_ms": compute_ms}
        if kind == "truncate":
            await asyncio.get_running_loop().run_in_executor(self._gpu, session.cache.truncate, request["length"])
            return reply | {"type": "ok"}
        if kind == "close_session":
            self.sessions.pop(request["session"], None)
            return reply | {"type": "ok"}
        return error(request, "unknown_type", f"unknown message type {kind!r}")

    def _run_stage(self, session: Session, request: dict, received: float):
        started = time.perf_counter()
        hidden = self.stage(request["hidden"], request["start_pos"], session.cache, request.get("per_layer_inputs"))
        hidden = hidden.cpu()  # also waits for the GPU to finish, so compute_ms is real
        done = time.perf_counter()
        return hidden, (started - received) * 1000, (done - started) * 1000

    async def _connection(self, ws: ServerConnection) -> None:
        try:
            async for frame in ws:
                request: dict = {}
                try:
                    request = decode(frame)
                    reply = await self.handle(request)
                except Exception as exc:  # report to the client instead of dropping the connection
                    log.exception("request failed")
                    reply = error(request, "internal", repr(exc))
                if self.delay_ms:
                    await asyncio.sleep(self.delay_ms / 1000)
                await ws.send(encode(reply))
        except ConnectionClosed:
            pass

    # --- background tasks ---

    async def _expire_sessions(self) -> None:
        while True:
            await asyncio.sleep(min(60.0, self.session_timeout))
            cutoff = time.monotonic() - self.session_timeout
            for sid in [s for s, sess in self.sessions.items() if sess.last_used < cutoff]:
                del self.sessions[sid]

    async def _heartbeat(self, tracker_url: str, gpu: str) -> None:
        info = {
            "peer_id": self.peer_id,
            "model": self.model,
            "start": self.stage.start,
            "end": self.stage.end,
            "url": self.url,
            "region": self.region,
            "gpu": gpu,
        }
        async with httpx.AsyncClient(base_url=tracker_url, timeout=5.0) as http:
            registered = False
            while True:
                try:
                    if not registered:
                        (await http.post("/register", json=info)).raise_for_status()
                        registered = True
                        log.info("registered with tracker %s", tracker_url)
                    else:
                        r = await http.post("/heartbeat", json={"peer_id": self.peer_id, "sessions": len(self.sessions)})
                        registered = r.status_code == 200  # 404: the tracker restarted and forgot us
                except httpx.HTTPError as exc:
                    log.warning("tracker unreachable: %s", exc)
                    registered = False
                await asyncio.sleep(HEARTBEAT_SECONDS)

    async def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        public_url: str | None = None,
        tracker_url: str | None = None,
        gpu: str = "",
        ready: asyncio.Event | None = None,
    ) -> None:
        """Serve until cancelled. Port 0 picks a free port; the URL is then in `self.url`."""
        if self.delay_ms and sys.platform == "win32":
            # Windows timers tick every 15.6 ms by default, which would round a 20 ms delay up to about 31 ms.
            import ctypes

            ctypes.windll.winmm.timeBeginPeriod(1)
        async with serve(self._connection, host, port, max_size=MAX_FRAME_BYTES, compression=None) as server:
            bound_port = server.sockets[0].getsockname()[1]
            self.url = public_url or f"ws://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{bound_port}"
            log.info("peer %s serving layers %d-%d at %s", self.peer_id, self.stage.start, self.stage.end - 1, self.url)
            tasks = [asyncio.create_task(self._expire_sessions())]
            if tracker_url:
                tasks.append(asyncio.create_task(self._heartbeat(tracker_url, gpu)))
            if ready is not None:
                ready.set()
            try:
                await asyncio.Future()  # run forever
            finally:
                for task in tasks:
                    task.cancel()
                self._gpu.shutdown(wait=False)
