"""Events a generating client sends to the tracker, for the dashboard.

By default they carry counts and timings only. With `share_text`, they also carry
the decoded text of each token, so the dashboard can show the token stream; that
reveals the output to whoever runs the tracker, so it is opt-in.

    generation_start  {mode, k, model}
    tokens            {mode, produced, accepted, proposed, draft_ms, verify_ms, text?}
    generation        {mode, k, tokens, seconds, tok_s, acceptance, tokens_per_trip}
"""

import time


class Reporter:
    def __init__(self, pipe, model: str, tokenizer=None, share_text: bool = False):
        self.pipe, self.model = pipe, model
        self.tokenizer = tokenizer if share_text else None
        self.mode, self.k = "plain", 0

    def _emit(self, kind: str, **fields) -> None:
        self.pipe.events.emit(
            {"type": kind, "time": time.time(), "client_id": self.pipe.client_id, "session": self.pipe.session, **fields}
        )

    def _text(self, tokens: list[int]) -> list[str] | None:
        if self.tokenizer is None:
            return None
        return [self.tokenizer.decode([t]) for t in tokens]

    def start(self, mode: str, k: int = 0) -> None:
        self.mode, self.k = mode, k
        self._emit("generation_start", mode=mode, k=k, model=self.model)

    def on_token(self, token: int) -> None:
        """Plain generation: one token per trip."""
        self._emit("tokens", mode="plain", produced=1, accepted=0, proposed=0, text=self._text([token]))

    def on_round(self, r, produced: list[int]) -> None:
        """Speculative generation: `r.accepted` guesses kept, then one token from the target."""
        self._emit(
            "tokens", mode="speculative", produced=len(produced), accepted=r.accepted, proposed=r.proposed,
            draft_ms=round(r.draft_ms, 2), verify_ms=round(r.verify_ms, 2), text=self._text(produced),
        )

    def finish(self, tokens: list[int], seconds: float, stats=None) -> None:
        self._emit(
            "generation", mode=self.mode, k=self.k, tokens=len(tokens), seconds=round(seconds, 3),
            tok_s=round(len(tokens) / seconds, 3) if seconds else 0.0,
            acceptance=round(stats.acceptance_rate, 4) if stats else None,
            tokens_per_trip=round(stats.tokens_per_round, 3) if stats else 1.0,
        )
