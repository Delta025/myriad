# Running a Myriad swarm on RunPod

This runs Gemma 4 31B across GPU pods in two regions, with the client on your own machine.

## Layout

| Where | Runs | Needs |
| --- | --- | --- |
| Your machine (client) | Embedding, layer 0, layers 58–59, output head, MTP drafter | About 6 GB of GPU memory (an RTX 3080 works) and about 7 GB of downloads |
| Pod A (region 1) | Tracker and dashboard, plus a peer serving layers 1–29 | A 48 GB GPU (L40S, A6000 or A40) |
| Pod B (region 2) | A peer serving layers 30–57 | A 48 GB GPU |

The client keeps layers 58–59 because Gemma 4's MTP drafter reads them. Each peer downloads only the tensors of its own layers, about 28 GB, by HTTP range requests from the Hugging Face Hub.

Peers talk to clients over **direct TCP ports**, not RunPod's HTTP proxy. The proxy is not documented to carry WebSockets and would add a hop to every measurement. The tracker, which is plain HTTP, sits behind the proxy.

## 1. Create the pods

For each pod, in the RunPod console (or through the RunPod MCP server):

- **Cloud: Secure Cloud.** Public IPs stay stable there; on Community Cloud they can change when a pod restarts.
- **GPU:** one 48 GB GPU. Pick data centers far enough apart to show real latency, for example one in North America and one in Europe.
- **Template:** a recent RunPod PyTorch image. Only the CUDA driver and a shell are needed; the setup script installs its own Python.
- **Disk:** container disk 20 GB, plus a volume of at least 60 GB mounted at `/workspace`. Weights are cached there, so a restart does not download them again.
- **Ports:** expose **TCP 9000** for the peer. On pod A also expose **HTTP 8000** for the tracker.

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

The dashboard is then at `https://<pod-A-id>-8000.proxy.runpod.net/`.

## 3. Start pod B: second peer

```bash
cd /workspace
curl -fsSL https://raw.githubusercontent.com/Delta025/myriad/main/scripts/pod_setup.sh -o pod_setup.sh
NODE_NAME=peer-b nohup bash pod_setup.sh peer google/gemma-4-31B-it 30:58 https://<pod-A-id>-8000.proxy.runpod.net > peer.log 2>&1 &
tail -f peer.log
```

A peer is ready when its log shows `registered with tracker`. The first start downloads its weights (about 28 GB).

The peer announces itself as `ws://$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_9000`, both taken from the pod's environment. Set `PUBLIC_URL` to override that.

## 4. Generate from your machine

```bash
uv run myriad generate google/gemma-4-31B-it --tracker https://<pod-A-id>-8000.proxy.runpod.net --first-layers 1 --last-layers 2 --mtp google/gemma-4-31B-it-assistant --k 2 --share-text --prompt "Why is the sky blue?"
```

Drop `--mtp …` to compare against plain decoding, or use `--draft google/gemma-4-E2B-it` to draft with E2B instead.

To benchmark plain decoding against speculative decoding over the real network:

```bash
uv run python scripts/bench_speculative.py google/gemma-4-31B-it --tracker https://<pod-A-id>-8000.proxy.runpod.net --first-layers 1 --last-layers 2 --mtp google/gemma-4-31B-it-assistant --k 2 4
```

## 5. Stop the pods

Pods bill while they run. **Stop** them when you are done: the `/workspace` volume and its cached weights are kept, and only storage is billed. **Terminate** them only if you no longer need the cached weights.

## Troubleshooting

- **A peer can't register:** check the tracker URL. From pod B it must be the proxy URL of pod A (`https://…-8000.proxy.runpod.net`), and the tracker must be running (`tracker.log`).
- **The client can't reach a peer:** check that TCP port 9000 is exposed and the pod has a public IP. The pod's Connect menu shows the public IP and port mapping.
- **CUDA unavailable:** the setup script installs a PyTorch build that matches the host's driver. If it still fails, check `nvidia-smi` on the pod.
- **Simulating extra latency:** set `DELAY_MS=50` before starting a peer. Or, if the pod allows it, use `scripts/netem.sh` (needs `NET_ADMIN`, often unavailable in containers).
