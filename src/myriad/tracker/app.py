"""The tracker: where peers announce themselves and clients find a route, as in early BitTorrent.

It never sees prompts or activations, only which peer serves which layers,
plus timing events for the dashboard.

GET  /                the live dashboard
POST /register        peer info: peer_id, model, start, end, url, region, gpu, num_layers
POST /heartbeat       {peer_id, sessions}; 404 if the tracker doesn't know the peer
GET  /peers           live peers
GET  /route           ?model=&start=&end=: peers whose layer ranges chain exactly from start to end
POST /events          a list of events (from clients)
GET  /events          recent events
WS   /events/stream   recent events, then live ones
"""

import asyncio
import random
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

DASHBOARD = Path(__file__).parent.parent / "dashboard" / "index.html"

PEER_TIMEOUT_SECONDS = 15.0  # three missed heartbeats
EVENT_HISTORY = 2000


class PeerInfo(BaseModel):
    peer_id: str
    model: str
    start: int
    end: int
    url: str
    region: str = ""
    gpu: str = ""
    num_layers: int = 0  # layers in the whole model, for the dashboard's coverage bar
    peer_key: str = ""  # the peer node's Ed25519 public key
    name: str = ""  # self-declared, for display only


class Heartbeat(BaseModel):
    peer_id: str
    sessions: int = 0
    queue: list[str] = []  # names of requesters waiting, in the order the peer would serve them
    credits: list[dict] = []  # the peer's own view: work received from / given to each counterparty


class Tracker:
    def __init__(self, peer_timeout: float = PEER_TIMEOUT_SECONDS):
        self.peer_timeout = peer_timeout
        self.peers: dict[str, dict] = {}
        self.events: deque[dict] = deque(maxlen=EVENT_HISTORY)
        self.subscribers: set[asyncio.Queue] = set()

    def live_peers(self) -> list[dict]:
        cutoff = time.time() - self.peer_timeout
        return [p for p in self.peers.values() if p["last_seen"] >= cutoff]

    def route(self, model: str, start: int, end: int) -> list[dict] | None:
        """Fewest peers whose ranges chain exactly from `start` to `end`; random among equals."""
        candidates = [p for p in self.live_peers() if p["model"] == model]
        random.shuffle(candidates)  # spread clients over replicas
        # Breadth-first over layer boundaries, so the first route found has the fewest hops.
        frontier, came_from = [start], {start: None}
        while frontier:
            next_frontier = []
            for position in frontier:
                for peer in candidates:
                    if peer["start"] == position and peer["end"] <= end and peer["end"] not in came_from:
                        came_from[peer["end"]] = (position, peer)
                        next_frontier.append(peer["end"])
            frontier = next_frontier
        if end not in came_from:
            return None
        hops, position = [], end
        while came_from[position] is not None:
            position, peer = came_from[position]
            hops.append(peer)
        return hops[::-1]

    def publish(self, events: list[dict]) -> None:
        for event in events:
            self.events.append(event)
            for queue in self.subscribers:
                queue.put_nowait(event)


def create_app(tracker: Tracker | None = None) -> FastAPI:
    # Endpoints that publish events are async: they must run on the event loop, where the
    # subscribers' asyncio queues live (plain `def` endpoints run in a thread pool).
    tracker = tracker or Tracker()
    app = FastAPI(title="Myriad tracker")
    app.state.tracker = tracker

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD.read_text(encoding="utf-8")

    @app.post("/register")
    async def register(info: PeerInfo):
        tracker.peers[info.peer_id] = info.model_dump() | {"last_seen": time.time(), "sessions": 0, "queue": [],
                                                           "credits": []}
        tracker.publish([{"type": "peer_joined", "time": time.time(), **info.model_dump()}])
        return {"ok": True}

    @app.post("/heartbeat")
    async def heartbeat(beat: Heartbeat):
        peer = tracker.peers.get(beat.peer_id)
        if peer is None:
            raise HTTPException(404, "unknown peer; register again")
        peer.update(last_seen=time.time(), sessions=beat.sessions, queue=beat.queue, credits=beat.credits)
        return {"ok": True}

    @app.get("/peers")
    def peers():
        return tracker.live_peers()

    @app.get("/route")
    def route(model: str, start: int, end: int):
        hops = tracker.route(model, start, end)
        if hops is None:
            raise HTTPException(404, f"no live peers cover layers {start}-{end - 1} of {model}")
        return hops

    @app.post("/events")
    async def post_events(events: list[dict]):
        tracker.publish(events)
        return {"ok": True}

    @app.get("/events")
    def get_events(limit: int = 200):
        return list(tracker.events)[-limit:]

    @app.websocket("/events/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        queue: asyncio.Queue = asyncio.Queue()
        tracker.subscribers.add(queue)
        try:
            for event in list(tracker.events)[-200:]:
                await ws.send_json(event)
            while True:
                await ws.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            tracker.subscribers.discard(queue)

    return app
