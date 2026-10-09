"""Start a tracker and peer processes on this machine, for checks and benchmarks.

Each peer is a real `myriad peer` process, as it would be on separate machines.
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

TRACKER_PORT, FIRST_PEER_PORT = 8765, 9765
TRACKER_URL = f"http://127.0.0.1:{TRACKER_PORT}"


class LocalSwarm:
    def __init__(self, model: str, split: list[tuple[int, int]], log_dir: Path, delay_ms: float = 0.0, timeout: float = 300):
        cli = [sys.executable, "-m", "myriad.cli"]
        self.log_dir = log_dir
        self.procs: list[subprocess.Popen] = []

        def launch(name, args):
            log = open(log_dir / f"{name}.log", "w")
            self.procs.append(subprocess.Popen(cli + args, stdout=log, stderr=subprocess.STDOUT))

        launch("tracker", ["tracker", "--host", "127.0.0.1", "--port", str(TRACKER_PORT)])
        for i, (a, b) in enumerate(split):
            launch(
                f"peer{i}",
                ["-v", "peer", model, "--layers", f"{a}:{b}", "--host", "127.0.0.1", "--port", str(FIRST_PEER_PORT + i),
                 "--tracker", TRACKER_URL, "--region", f"local-{i}", "--delay-ms", str(delay_ms)],
            )

        deadline = time.monotonic() + timeout
        try:
            while True:
                if any(p.poll() is not None for p in self.procs):
                    raise RuntimeError(f"a swarm process exited early; see logs in {log_dir}")
                try:
                    if len(httpx.get(f"{TRACKER_URL}/peers", timeout=2).json()) == len(split):
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"peers did not register; see logs in {log_dir}")
                time.sleep(0.5)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
