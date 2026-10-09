import time

from fastapi.testclient import TestClient

from myriad.tracker.app import Tracker, create_app


def peer(pid, start, end, model="m"):
    return {"peer_id": pid, "model": model, "start": start, "end": end, "url": f"ws://{pid}", "region": "x"}


def client_with(*peers, timeout=15.0):
    tracker = Tracker(peer_timeout=timeout)
    client = TestClient(create_app(tracker))
    for p in peers:
        assert client.post("/register", json=p).status_code == 200
    return client, tracker


def ids(route):
    return [hop["peer_id"] for hop in route]


def test_route_chains_layer_ranges_with_fewest_hops():
    client, _ = client_with(peer("a", 1, 5), peer("b", 5, 9), peer("c", 9, 12), peer("big", 1, 9), peer("other", 5, 9, model="n"))
    route = client.get("/route", params={"model": "m", "start": 1, "end": 12}).json()
    assert ids(route) == ["big", "c"]


def test_no_route_when_layers_are_missing():
    client, _ = client_with(peer("a", 1, 5), peer("c", 6, 12))
    assert client.get("/route", params={"model": "m", "start": 1, "end": 12}).status_code == 404


def test_silent_peers_drop_out_and_heartbeats_keep_them():
    client, tracker = client_with(peer("a", 0, 4), peer("b", 0, 4), timeout=10)
    tracker.peers["a"]["last_seen"] = time.time() - 60
    assert [p["peer_id"] for p in client.get("/peers").json()] == ["b"]
    assert client.post("/heartbeat", json={"peer_id": "a"}).status_code == 200
    assert len(client.get("/peers").json()) == 2
    assert client.post("/heartbeat", json={"peer_id": "zzz"}).status_code == 404


def test_events_are_stored_and_streamed():
    client, _ = client_with()
    with client.websocket_connect("/events/stream") as ws:
        client.post("/events", json=[{"type": "forward", "n": 1}])
        assert ws.receive_json() == {"type": "forward", "n": 1}
    assert client.get("/events").json()[-1] == {"type": "forward", "n": 1}
