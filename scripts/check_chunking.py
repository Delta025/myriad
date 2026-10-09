"""Check that verifying several tokens in one call gives the same logits as one token at a time.

    uv run python scripts/check_chunking.py google/gemma-4-E2B-it --dtype float32
    uv run python scripts/check_chunking.py google/gemma-4-E2B-it --dtype bfloat16

Speculative decoding sends the last token plus k guesses in one call. The maths
is the same as k + 1 single-token calls, but GPU kernels sum in a different
order for different chunk sizes. In float32 the logits should agree to rounding
error. In bfloat16 they differ by a few ulps, which can flip greedy choices
where the top two logits are tied within that noise; this reports where.
"""

import argparse

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from myriad.client.generate import greedy_generate
from myriad.model.checkpoint import Checkpoint
from myriad.model.pipeline import LocalPipeline

PROMPT = "Write a short paragraph about how bees communicate."
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--chunks", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    ckpt = Checkpoint(args.model)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=True, return_dict=False
    )
    pipe = LocalPipeline.from_checkpoint(ckpt, None, DTYPES[args.dtype], args.device, "cpu")
    seq = prompt + greedy_generate(pipe, prompt, args.tokens)
    p, n = len(prompt), args.tokens - 1  # positions p-1 .. p+n-2 predict the generated tokens

    pipe.truncate(0)
    single = [pipe.forward(prompt, 0)[-1:]]
    for i in range(n - 1):
        single.append(pipe.forward([seq[p + i]], p + i))
    single = torch.cat(single)
    top2 = single.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]

    ok = True
    for chunk in args.chunks:
        pipe.truncate(0)
        pipe.forward(prompt[:-1], 0)
        out, pos = [], p - 1
        while pos < p - 1 + n:
            size = min(chunk, p - 1 + n - pos)
            out.append(pipe.forward(seq[pos : pos + size], pos))
            pos += size
        chunked = torch.cat(out)
        diff = (chunked - single).abs().amax(dim=-1)
        flips = (chunked.argmax(-1) != single.argmax(-1)).nonzero().flatten().tolist()
        print(f"chunks of {chunk}: max |logit difference| {diff.max():.4f} (median {diff.median():.4f}), "
              f"greedy choice changes at {len(flips)} of {n} positions"
              + (f", where the top-2 margin was {[round(margin[i].item(), 3) for i in flips]}" if flips else ""))
        ok &= not flips
    print(f"median top-2 margin: {margin.median():.2f}")
    print("PASS: no greedy choice depends on chunk size" if ok else "NOTE: some greedy choices depend on chunk size (see margins)")


if __name__ == "__main__":
    main()
