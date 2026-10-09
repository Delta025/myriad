"""Run a tracker and peer processes on this machine until Ctrl+C, for demos and dashboard work.

    uv run python scripts/local_swarm.py google/gemma-4-E2B-it --peers 2 --delay-ms 50

The dashboard is then at http://127.0.0.1:8765/. Generate against it with:

    uv run myriad generate google/gemma-4-E2B-it --tracker http://127.0.0.1:8765 --share-text --prompt "..."
"""

import argparse
import tempfile
import time
from pathlib import Path

from myriad.localswarm import TRACKER_URL, LocalSwarm
from myriad.model.checkpoint import Checkpoint
from myriad.model.split import even_split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--peers", type=int, default=2)
    parser.add_argument("--first-layers", type=int, default=1, help="layers the client will run before the swarm")
    parser.add_argument("--last-layers", type=int, default=0, help="layers the client will run after the swarm")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="simulated round trip added by each peer")
    args = parser.parse_args()

    config = Checkpoint(args.model).text_config()
    split = even_split(config, args.peers, start=args.first_layers, end=config.num_hidden_layers - args.last_layers)
    log_dir = Path(tempfile.mkdtemp(prefix="myriad-swarm-"))
    with LocalSwarm(args.model, split, log_dir, args.delay_ms):
        print(f"peers {split} are up; dashboard at {TRACKER_URL}/  (logs in {log_dir}; Ctrl+C to stop)", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
