"""Check that a split pipeline matches stock Transformers on a real Gemma 4 model.

    uv run python scripts/check_equivalence.py google/gemma-4-E2B-it --stages 3
    uv run python scripts/check_equivalence.py google/gemma-4-E4B-it --stages 3 --ends-device cpu

Phase 1 runs the Transformers reference (greedy `generate`, keeping the logits
of every step). Phase 2 frees it and
runs our pipeline with the same device for every part (decoder layers on
`--stage-device`, embedding, per-layer embeddings and head on `--ends-device`),
so both compute each layer on the same hardware.

PASS means identical greedy tokens. The max logit difference is reported too;
with the same devices it is expected to be 0.0 (measured on E2B and E4B, 2026-10-09).
Do not compare against a single uncached forward over the whole text: in bf16 that
differs from Transformers' own cached generation by about 1.0 already.
"""

import argparse
import gc
import time

import torch
from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration
from transformers.utils import logging as hf_logging

from myriad.model.checkpoint import Checkpoint
from myriad.model.pipeline import LocalPipeline
from myriad.model.split import even_split

PROMPT = "Explain in three sentences why the sky is blue."


def gib(n_bytes: float) -> str:
    return f"{n_bytes / 2**30:.2f} GiB"


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(to_device(v, device) for v in x)
    return x


def move_inputs_hook(device):
    def hook(module, args, kwargs):
        return to_device(args, device), {k: to_device(v, device) for k, v in kwargs.items()}

    return hook


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def run_reference(ckpt: Checkpoint, prompt_ids: list[int], n_new: int, stage_device: str, ends_device: str):
    config = AutoConfig.from_pretrained(ckpt.path)
    config.vision_config = None  # text only: skip loading the vision and audio towers
    config.audio_config = None
    # Load on CPU, then move only the decoder layers. (Accelerate's device_map leaves CPU-mapped
    # weights on the meta device, which Gemma 4's embedding code reads directly.)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ckpt.path, config=config, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    model.to(ends_device)
    for layer in model.model.language_model.layers:
        layer.to(stage_device)
        layer.register_forward_pre_hook(move_inputs_hook(stage_device), with_kwargs=True)
    model.model.language_model.norm.register_forward_pre_hook(move_inputs_hook(ends_device), with_kwargs=True)
    # Rotary cos/sin must be computed on the layers' device too: CPU and CUDA round some
    # positions differently (e.g. position 310 on E2B's global layers), as our stages do it on their device.
    rotary = model.model.language_model.rotary_emb.to(stage_device)
    rotary.register_forward_pre_hook(move_inputs_hook(stage_device), with_kwargs=True)
    print(f"  loaded; GPU memory {gib(torch.cuda.memory_allocated())}")

    t = time.perf_counter()
    ids = torch.tensor([prompt_ids], device=ends_device)
    out = model.generate(
        ids, max_new_tokens=n_new, do_sample=False, eos_token_id=None, pad_token_id=0,
        output_logits=True, return_dict_in_generate=True,
    )
    generated = out.sequences[0, len(prompt_ids) :].tolist()
    print(f"  generated {len(generated)} tokens in {time.perf_counter() - t:.1f}s")
    logits = torch.cat([step.float().cpu() for step in out.logits])  # one row per generated token
    del model
    return generated, logits


def run_pipeline(ckpt: Checkpoint, prompt_ids: list[int], n_new: int, split, stage_device: str, ends_device: str):
    pipe = LocalPipeline.from_checkpoint(ckpt, split, torch.bfloat16, stage_device, ends_device)
    print(f"  loaded; GPU memory {gib(torch.cuda.memory_allocated())}")

    # Same calls as myriad.client.generate.greedy_generate, keeping each step's logits for the comparison.
    t = time.perf_counter()
    logits = [pipe.forward(prompt_ids, start_pos=0)[-1:]]
    generated = [int(logits[-1][-1].argmax())]
    while len(generated) < n_new:
        logits.append(pipe.forward([generated[-1]], start_pos=len(prompt_ids) + len(generated) - 1))
        generated.append(int(logits[-1][-1].argmax()))
    print(f"  generated {len(generated)} tokens in {time.perf_counter() - t:.1f}s")
    del pipe
    return generated, torch.cat(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face repo id or local checkpoint directory")
    parser.add_argument("--stages", type=int, default=3, help="number of stages to split the layers into")
    parser.add_argument("--tokens", type=int, default=48, help="tokens to generate")
    parser.add_argument("--stage-device", default="cuda")
    parser.add_argument("--ends-device", default="cpu", help="device for embedding, per-layer embeddings and head")
    args = parser.parse_args()
    hf_logging.set_verbosity_error()  # the vision/audio weights we skip are reported as unexpected
    hf_logging.disable_progress_bar()

    ckpt = Checkpoint(args.model)
    config = ckpt.text_config()
    split = even_split(config, args.stages)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.path)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    print(f"{args.model}: {config.num_hidden_layers} layers, split {split}, prompt {len(prompt_ids)} tokens")

    print("Phase 1: Transformers reference")
    ref_tokens, ref_logits = run_reference(ckpt, prompt_ids, args.tokens, args.stage_device, args.ends_device)
    free_memory()

    print("Phase 2: Myriad pipeline")
    our_tokens, our_logits = run_pipeline(ckpt, prompt_ids, args.tokens, split, args.stage_device, args.ends_device)

    print("\nReference:", repr(tokenizer.decode(ref_tokens)))
    print("Pipeline: ", repr(tokenizer.decode(our_tokens)))
    first_diff = next((i for i, (a, b) in enumerate(zip(ref_tokens, our_tokens)) if a != b), None)
    max_diff = (our_logits - ref_logits).abs().max().item()
    print(f"max |logit difference| vs Transformers over {ref_logits.shape[0]} steps: {max_diff:.4f}")
    if first_diff is None:
        print(f"PASS: {len(ref_tokens)} greedy tokens identical")
    else:
        print(f"FAIL: tokens differ from position {first_diff}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
