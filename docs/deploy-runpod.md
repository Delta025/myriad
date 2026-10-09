# Running a Myriad swarm on RunPod

This runs Gemma 4 31B across GPU pods in two regions. The client runs on your own machine, or on a third pod driven from any browser.

## Layout

| Where | Runs | Needs |
| --- | --- | --- |
| Your machine (client) | Embedding, layer 0, layers 58–59, output head, MTP drafter | About 4 GB of GPU memory (an RTX 3080 works) and about 7 GB of downloads |
| Pod A (region 1) | Tracker and dashboard, plus a peer serving layers 1–29 | A 48 GB GPU (L40S, A6000 or A40) |
| Pod B (region 2) | A peer serving layers 30–57 | A 48 GB GPU |
| Pod C (optional) | The client, as an HTTP service for the dashboard's "Ask" box | Any GPU with 16 GB or more, **in a data center with no peer** |

The client keeps layers 58–59 because Gemma 4's MTP drafter reads them. Each peer downloads only the tensors of its own layers, about 28 GB, by HTTP range requests from the Hugging Face Hub.

Peers talk to clients over **direct TCP ports**, not RunPod's HTTP proxy. The proxy is not documented to carry WebSockets and would add a hop to every measurement. The tracker and the client service, both plain HTTP, sit behind the proxy.

## 1. Create the pods

For each pod, in the RunPod console (or through the RunPod MCP server):

- **Cloud: Secure Cloud.** Public IPs stay stable there; on Community Cloud they can change when a pod restarts.
- **GPU:** one 48 GB GPU for each peer. Pick data centers far enough apart to show real latency, for example one in North America and one in Europe.
- **Template:** a recent RunPod PyTorch image. Only the CUDA driver and a shell are needed; the setup script installs its own Python and a PyTorch build that matches the host's driver.
- **Disk:** container disk 20 GB, plus a volume mounted at `/workspace`: at least 60 GB for a peer, 30 GB for a client. Weights are cached there, so a restart does not download them again.
- **Ports:** expose **TCP 9000** on each peer, **HTTP 8000** on pod A (tracker), and **HTTP 8001** on pod C (client service).

Each step below can also be the pod's start command (`bash -c "…"`), so the pod sets itself up with no terminal.

## 2. Start pod A: tracker and first peer

Open a terminal on the pod (web terminal or SSH) and run:

```bash
cd /workspace
curl -fsSL https://raw.githubusercontent.com/Delta025/myriad/main/scripts/pod_setup.sh -o pod_setup.sh
nohup bash pod_setup.sh tracker > tracker.log 2>&1 &
until curl -sf http://localhost:8000/peers > /dev/null; do sleep 3; done   # setup done, tracker up
NODE_NAME=peer-a nohup bash pod_setup.sh peer google/gemma-4-31B-it 1:30 http://localhost:8000 > peer.log 2>&1 &
tail -f peer.log
```

The tracker's address is `https://POD_A_ID-8000.proxy.runpod.net`, where `POD_A_ID` is pod A's ID from the console. The commands below read it from a variable; set it once in each shell:

```bash
TRACKER=https://POD_A_ID-8000.proxy.runpod.net   # replace POD_A_ID
```

The dashboard is at `$TRACKER/`.

## 3. Start pod B: second peer

```bash
cd /workspace
curl -fsSL https://raw.githubusercontent.com/Delta025/myriad/main/scripts/pod_setup.sh -o pod_setup.sh
NODE_NAME=peer-b nohup bash pod_setup.sh peer google/gemma-4-31B-it 30:58 "$TRACKER" > peer.log 2>&1 &
tail -f peer.log
```

A peer is ready when its log shows `registered with tracker`. The first start downloads its weights (about 28 GB).

The peer announces itself as `ws://$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_9000`, both taken from the pod's environment. Set `PUBLIC_URL` to override that. The port changes when a pod restarts; the peer re-registers on its own.

## 4. Generate from your machine

```bash
uv run myriad generate google/gemma-4-31B-it --tracker "$TRACKER" --first-layers 1 --last-layers 2 --mtp google/gemma-4-31B-it-assistant --k 2 --share-text --prompt "Why is the sky blue?"
```

Drop `--mtp …` to compare against plain decoding, or use `--draft google/gemma-4-E2B-it` to draft with E2B instead.

To benchmark plain decoding against speculative decoding over the real network:

```bash
uv run python scripts/bench_speculative.py google/gemma-4-31B-it --tracker "$TRACKER" --first-layers 1 --last-layers 2 --mtp google/gemma-4-31B-it-assistant --k 2 4
```

## 5. Optional: a client pod, driven from any browser

For a demo from a machine that cannot run the client (an old laptop, a phone), run the client on pod C as an HTTP service:

```bash
cd /workspace
curl -fsSL https://raw.githubusercontent.com/Delta025/myriad/main/scripts/pod_setup.sh -o pod_setup.sh
NODE_NAME=demo-client nohup bash pod_setup.sh client google/gemma-4-31B-it "$TRACKER" google/gemma-4-31B-it-assistant > client.log 2>&1 &
```

**Put pod C in a data center where no peer runs.** A pod cannot reach the public address of another pod in the same data center (connections are refused), while machines outside it can.

When `https://POD_C_ID-8001.proxy.runpod.net/health` answers, open the dashboard with the client's address. An "Ask the swarm" box appears, with a switch for speculative decoding:

```
https://POD_A_ID-8000.proxy.runpod.net/?client=https://POD_C_ID-8001.proxy.runpod.net
```

## 6. Stop the pods

Pods bill while they run. **Stop** them when you are done: the `/workspace` volume and its cached weights are kept, and only storage is billed. **Terminate** them only if you no longer need the cached weights.

## Troubleshooting

- **A peer can't register:** check the tracker URL. From another pod it must be the proxy URL of pod A (`https://…-8000.proxy.runpod.net`), and the tracker must be running (`tracker.log`). A `502` from the proxy means nothing is listening yet.
- **The client can't reach a peer** ("connection refused"): check that TCP port 9000 is exposed and the pod has a public IP; the pod's Connect menu shows the mapping. If the client is itself a pod, make sure it is not in the same data center as that peer.
- **"Could not reach the client" in the Ask box:** the client service is not up yet (check `https://POD_C_ID-8001.proxy.runpod.net/health`) or failed to start (check `client.log`).
- **CUDA unavailable:** the setup script installs a PyTorch build the host's driver supports (for example CUDA 12.6 on a 12.8 driver). If it still fails, check `nvidia-smi` on the pod.
- **Simulating extra latency:** set `DELAY_MS=50` before starting a peer. Or, if the pod allows it, use `scripts/netem.sh` (needs `NET_ADMIN`, often unavailable in containers).
