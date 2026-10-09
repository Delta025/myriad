"""The order in which a peer runs waiting requests: tit-for-tat with an optimistic slot.

When several requests wait for the GPU, the peer runs first the one whose
requester has done the most work for this node (from its ledger). Its own
client always comes first. With probability `unchoke`, the slot instead goes to
a random waiting request, so newcomers and freeloaders still make progress
(BitTorrent's "optimistic unchoke").

Without a ledger every requester scores 0, which is first come, first served.
"""

import asyncio
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass
class _Job:
    requester: str
    arrived: int
    fn: Callable
    future: asyncio.Future = field(default=None)


class Scheduler:
    def __init__(self, score: Callable[[str], float] | None = None, unchoke: float = 0.2, seed: int | None = None):
        self.score = score or (lambda requester: 0.0)
        self.unchoke = unchoke
        self._rng = random.Random(seed)
        self._waiting: list[_Job] = []
        self._arrivals = 0
        self._wakeup: asyncio.Event | None = None
        self._worker: asyncio.Task | None = None
        self._gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage")
        self.picks: list[tuple[str, bool]] = []  # (requester, by optimistic unchoke), for tests and the dashboard

    async def run(self, requester: str, fn: Callable):
        """Run `fn` on the GPU thread when this request's turn comes; returns its result."""
        if self._worker is None:
            self._wakeup = asyncio.Event()
            self._worker = asyncio.create_task(self._work())
        self._arrivals += 1
        job = _Job(requester, self._arrivals, fn, asyncio.get_running_loop().create_future())
        self._waiting.append(job)
        self._wakeup.set()
        return await job.future

    def queue(self) -> list[str]:
        """Requesters waiting, in the order they would be served without unchoking."""
        return [job.requester for job in sorted(self._waiting, key=self._rank)]

    def _rank(self, job: _Job):
        return (-self.score(job.requester), job.arrived)

    def _pick(self) -> _Job:
        optimistic = len(self._waiting) > 1 and self._rng.random() < self.unchoke
        job = self._rng.choice(self._waiting) if optimistic else min(self._waiting, key=self._rank)
        self._waiting.remove(job)
        self.picks.append((job.requester, optimistic))
        del self.picks[:-1000]
        return job

    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            while not self._waiting:
                self._wakeup.clear()
                await self._wakeup.wait()
            # Let requests that arrive in the same instant join the choice.
            await asyncio.sleep(0)
            job = self._pick()
            if job.future.done():  # the requester went away while waiting
                continue
            try:
                result = await loop.run_in_executor(self._gpu, job.fn)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._gpu.shutdown(wait=False)
