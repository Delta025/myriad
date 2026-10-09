"""A client as a small HTTP service: ask a question from a browser, the swarm answers.

The service plays the role of the user's PC. It loads the client's parts of
the model (embedding, first and last layers, head) and the drafter once, then
each request opens a session on the swarm and generates, exactly like
`myriad generate`. The dashboard's "Ask" box calls it, so a demo can be driven
from any browser, for example a laptop that could never run the model itself.

    POST /generate  {"prompt": "...", "speculative": true, "k": 2, "max_tokens": 80}
    GET  /health
"""

import logging
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from myriad.client.drafters import MTPDrafter
from myriad.client.generate import generate
from myriad.client.remote import RemotePipeline
from myriad.client.reporting import Reporter
from myriad.client.sampling import GREEDY
from myriad.client.speculative import speculative_generate
from myriad.model.checkpoint import Checkpoint

log = logging.getLogger("myriad.client.service")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    speculative: bool = True
    k: int = Field(2, ge=1, le=8)
    max_tokens: int = Field(80, ge=1, le=256)


class ClientService:
    def __init__(self, model: str, tracker_url: str, first_layers: int, last_layers: int, mtp: str | None,
                 device: str = "cuda", identity=None, ledger=None, share_text: bool = True):
        from transformers import AutoTokenizer, Gemma4AssistantForCausalLM, GenerationConfig

        self.model, self.tracker_url = model, tracker_url
        self.first_layers, self.last_layers = first_layers, last_layers
        self.identity, self.ledger, self.share_text = identity, ledger, share_text
        self.checkpoint = Checkpoint(model)
        self.parts = RemotePipeline.load_client_parts(self.checkpoint, first_layers, last_layers, torch.bfloat16,
                                                      device, "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint.path)
        eos = GenerationConfig.from_pretrained(self.checkpoint.path).eos_token_id
        self.stop = eos if isinstance(eos, list) else [eos]
        self.assistant = (Gemma4AssistantForCausalLM.from_pretrained(mtp, dtype=torch.bfloat16).to(device).eval()
                          if mtp else None)
        self._busy = threading.Lock()  # one generation at a time: it uses the whole client

    def generate(self, req: GenerateRequest) -> dict:
        if not self._busy.acquire(blocking=False):
            raise HTTPException(409, "already answering a question; try again in a moment")
        try:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": req.prompt}], add_generation_prompt=True, tokenize=True, return_dict=False
            )
            pipe = RemotePipeline.connect(self.checkpoint, self.tracker_url, model=self.model,
                                          first_layers=self.first_layers, last_layers=self.last_layers,
                                          parts=self.parts, identity=self.identity, ledger=self.ledger)
            reporter = Reporter(pipe, self.model, self.tokenizer, share_text=self.share_text)
            speculative = req.speculative and self.assistant is not None
            with pipe:
                t = time.perf_counter()
                stats = None
                if speculative:
                    reporter.start("speculative", req.k)
                    tokens, stats = speculative_generate(pipe, MTPDrafter(pipe, self.assistant), prompt,
                                                         req.max_tokens, req.k, GREEDY, self.stop,
                                                         on_round=reporter.on_round)
                else:
                    reporter.start("plain")
                    tokens = generate(pipe, prompt, req.max_tokens, GREEDY, self.stop, on_token=reporter.on_token)
                seconds = time.perf_counter() - t
                reporter.finish(tokens, seconds, stats)
            return {
                "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                "tokens": len(tokens),
                "seconds": round(seconds, 2),
                "tok_s": round(len(tokens) / seconds, 2),
                "mode": f"speculative, k={req.k}" if speculative else "plain",
                "acceptance": round(stats.acceptance_rate, 3) if stats else None,
                "tokens_per_trip": round(stats.tokens_per_round, 2) if stats else 1.0,
            }
        finally:
            self._busy.release()


def create_app(service: ClientService) -> FastAPI:
    app = FastAPI(title="Myriad client")
    # The dashboard is served by the tracker, on another origin, and calls this service directly.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/health")
    def health():
        return {"ok": True, "model": service.model, "drafter": service.assistant is not None,
                "busy": service._busy.locked()}

    @app.post("/generate")
    def generate_endpoint(req: GenerateRequest):  # plain def: FastAPI runs it in a worker thread
        try:
            return service.generate(req)
        except HTTPException:
            raise
        except Exception as exc:
            # An unhandled error would come back without CORS headers, and the browser would only say
            # "CORS". Report what failed (for example a peer that cannot be reached) instead.
            log.exception("generation failed")
            raise HTTPException(502, f"generation failed: {type(exc).__name__}: {exc}") from exc

    return app
