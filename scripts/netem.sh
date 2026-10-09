#!/usr/bin/env bash
# Add real network latency on a Linux peer with tc netem (needs root or CAP_NET_ADMIN).
# Containers such as RunPod pods usually lack that; use `myriad peer --delay-ms` there instead.
#
#   sudo scripts/netem.sh add eth0 25ms 5ms   # 25 ms +/- 5 ms on every packet leaving eth0
#   sudo scripts/netem.sh show eth0
#   sudo scripts/netem.sh del eth0
#
# Delay applies to outgoing packets only, so a round trip through this peer gains it once.
set -euo pipefail

action=${1:?usage: netem.sh add|del|show <interface> [delay] [jitter]}
dev=${2:?interface, e.g. eth0}

case "$action" in
  add)  tc qdisc replace dev "$dev" root netem delay "${3:?delay, e.g. 25ms}" "${4:-0ms}" ;;
  del)  tc qdisc del dev "$dev" root ;;
  show) tc qdisc show dev "$dev" ;;
  *)    echo "unknown action: $action" >&2; exit 1 ;;
esac
