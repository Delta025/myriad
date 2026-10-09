#!/usr/bin/env bash
# Set up and start a Myriad tracker or peer on a fresh Linux GPU machine (made for RunPod pods).
#
#   bash pod_setup.sh tracker                                # tracker + dashboard on port 8000
#   bash pod_setup.sh peer google/gemma-4-31B-it 1:30 https://<tracker>   # serve layers 1-29
#
# Environment (all optional):
#   MYRIAD_REPO     git URL of the code      (default: https://github.com/Delta025/myriad)
#   MYRIAD_DIR      where to clone it        (default: /workspace/myriad)
#   PEER_PORT       peer port inside the pod (default: 9000; expose it as a TCP port)
#   PUBLIC_URL      URL clients use to reach this peer. On RunPod it is derived from
#                   RUNPOD_PUBLIC_IP and RUNPOD_TCP_PORT_<PEER_PORT>.
#   REGION, NODE_NAME, DELAY_MS    passed to `myriad peer`
#
# Weights go to /workspace (the pod's volume) so a restart does not download them again,
# and a peer downloads only the tensors of its own layers.
set -euo pipefail

role=${1:?usage: pod_setup.sh tracker | peer MODEL START:END TRACKER_URL}
MYRIAD_REPO=${MYRIAD_REPO:-https://github.com/Delta025/myriad}
MYRIAD_DIR=${MYRIAD_DIR:-/workspace/myriad}
PEER_PORT=${PEER_PORT:-9000}
export HF_HOME=${HF_HOME:-/workspace/hf}
export MYRIAD_CACHE=${MYRIAD_CACHE:-/workspace/myriad-cache}
# uv's cache stays on the container disk: on some RunPod hosts, moving files inside a cache on the
# /workspace volume fails with "Cross-device link" (the RunPod image points UV_CACHE_DIR there).
export UV_CACHE_DIR=/root/.cache/uv
export UV_LINK_MODE=copy

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ -d "$MYRIAD_DIR/.git" ]; then
  git -C "$MYRIAD_DIR" pull --ff-only
else
  git clone "$MYRIAD_REPO" "$MYRIAD_DIR"
fi
cd "$MYRIAD_DIR"
uv sync --no-dev

# PyPI's torch wheel targets a recent CUDA driver. If this host's driver is older, use a
# build for an older CUDA from the PyTorch index instead.
if command -v nvidia-smi >/dev/null && ! uv run --no-sync python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  for index in cu130 cu128 cu126 cu124; do
    if [ "${cuda:-0}" -ge "${index#cu}" ]; then break; fi
  done
  version=$(uv run --no-sync python -c "import torch; print(torch.__version__.split('+')[0])")
  echo "driver supports CUDA ${cuda}; installing torch ${version} for ${index}"
  uv pip install --reinstall "torch==${version}" --index-url "https://download.pytorch.org/whl/${index}"
  uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CUDA still unavailable'"
fi

case "$role" in
  tracker)
    exec uv run --no-sync myriad tracker --host 0.0.0.0 --port 8000
    ;;
  peer)
    model=${2:?model}; layers=${3:?START:END}; tracker=${4:?tracker URL}
    if [ -z "${PUBLIC_URL:-}" ]; then
      mapped_var="RUNPOD_TCP_PORT_${PEER_PORT}"
      if [ -n "${RUNPOD_PUBLIC_IP:-}" ] && [ -n "${!mapped_var:-}" ]; then
        PUBLIC_URL="ws://${RUNPOD_PUBLIC_IP}:${!mapped_var}"
      else
        echo "set PUBLIC_URL, or expose TCP port ${PEER_PORT} on a pod with a public IP" >&2
        exit 1
      fi
    fi
    echo "serving ${model} layers ${layers} at ${PUBLIC_URL}, tracker ${tracker}"
    exec uv run --no-sync myriad -v peer "$model" --layers "$layers" --host 0.0.0.0 --port "$PEER_PORT" \
      --public-url "$PUBLIC_URL" --tracker "$tracker" --region "${REGION:-${RUNPOD_DC_ID:-$(hostname)}}" \
      --identity /workspace/node --node-name "${NODE_NAME:-$(hostname)}" --delay-ms "${DELAY_MS:-0}"
    ;;
  *)
    echo "unknown role: $role" >&2; exit 1
    ;;
esac
