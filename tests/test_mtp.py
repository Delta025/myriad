"""M4b: the MTP drafter reads exactly what Transformers feeds it, and speculative output stays the target's."""

import pytest
import torch
from transformers import Gemma4AssistantForCausalLM, Gemma4ForConditionalGeneration

from myriad.client.drafters import MTPDrafter
from myriad.client.generate import greedy_generate
from myriad.client.remote import RemotePipeline
from myriad.client.sampling import GREEDY, Sampling
from myriad.client.speculative import speculative_generate
from myriad.model.checkpoint import Checkpoint
from myriad.model.pipeline import LocalPipeline
from myriad.model.split import last_layer_of_each_type
from myriad.testing import TINY_VOCAB, ThreadedSwarm, make_tiny_assistant


def tokens(n, seed=0):
    return torch.randint(0, TINY_VOCAB, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def variant(checkpoint):
    return "ple" if Checkpoint(checkpoint).text_config().hidden_size_per_layer_input else "dense"


@pytest.fixture
def assistant(tiny_checkpoint, tmp_path):
    path = make_tiny_assistant(tmp_path / "assistant", variant(tiny_checkpoint))
    return Gemma4AssistantForCausalLM.from_pretrained(path, dtype=torch.float32)


def test_source_layers():
    from transformers import Gemma4TextConfig

    assert last_layer_of_each_type(Gemma4TextConfig(num_hidden_layers=60)) == {"sliding_attention": 58, "full_attention": 59}
    e4b = Gemma4TextConfig(num_hidden_layers=42, num_kv_shared_layers=18)
    assert last_layer_of_each_type(e4b) == {"sliding_attention": 22, "full_attention": 23}


def test_first_guesses_match_transformers_assisted_generation(tiny_checkpoint, assistant):
    """Reproduce the first round of Transformers' Gemma4 assistant candidate generator and compare."""
    prompt, k = tokens(9, seed=1), 4
    target = Gemma4ForConditionalGeneration.from_pretrained(tiny_checkpoint, dtype=torch.float32, attn_implementation="sdpa")
    with torch.no_grad():
        out = target(input_ids=torch.tensor([prompt]), output_hidden_states=True, return_shared_kv_states=True)
        first = int(out.logits[0, -1].argmax())
        hidden, kv = out.hidden_states[-1][:, -1:], out.shared_kv_states
        embed = target.get_input_embeddings()
        expected, token = [], first
        for _ in range(k):
            inputs = torch.cat([embed(torch.tensor([[token]])), hidden], dim=-1)
            step = assistant(inputs_embeds=inputs, position_ids=torch.tensor([[len(prompt)]]), shared_kv_states=kv)
            token, hidden = int(step.logits[0, -1].argmax()), step.last_hidden_state
            expected.append(token)

    pipe = LocalPipeline.from_checkpoint(str(tiny_checkpoint), dtype=torch.float32)
    assert int(pipe.forward(prompt, 0)[-1].argmax()) == first
    guesses, _ = MTPDrafter(pipe, assistant).propose(prompt + [first], k, GREEDY)
    assert guesses == expected


@pytest.mark.parametrize("k", [1, 3])
def test_speculative_greedy_with_mtp_matches_plain(tiny_checkpoint, assistant, k):
    prompt = tokens(7, seed=2)
    expected = greedy_generate(LocalPipeline.from_checkpoint(str(tiny_checkpoint), dtype=torch.float32), prompt, 24)
    target = LocalPipeline.from_checkpoint(str(tiny_checkpoint), dtype=torch.float32)
    out, stats = speculative_generate(target, MTPDrafter(target, assistant), prompt, 24, k=k)
    assert out == expected
    assert stats.rounds  # it ran rounds, with rollbacks (a random drafter is often wrong)


def test_mtp_sampling_runs(tiny_checkpoint, assistant):
    target = LocalPipeline.from_checkpoint(str(tiny_checkpoint), dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    out, _ = speculative_generate(target, MTPDrafter(target, assistant), tokens(5, seed=3), 20, k=3,
                                  sampling=Sampling(temperature=1.0), generator=g)
    assert len(out) == 20


def test_mtp_over_the_network(tiny_checkpoint, assistant):
    # The client must run the drafter's source layers itself: on the dense variant the last two
    # layers (6, 7); on the ple variant its whole KV-sharing block (2-7).
    ple = variant(tiny_checkpoint) == "ple"
    peers, last = ([(1, 2)], 6) if ple else ([(1, 3), (3, 6)], 2)
    prompt = tokens(7, seed=4)
    expected = greedy_generate(LocalPipeline.from_checkpoint(str(tiny_checkpoint), dtype=torch.float32), prompt, 20)
    with ThreadedSwarm(tiny_checkpoint, peers, model="tiny") as swarm:
        target = RemotePipeline.connect(str(tiny_checkpoint), swarm.tracker_url, model="tiny", first_layers=1,
                                        last_layers=last, dtype=torch.float32)
        with target:
            out, _ = speculative_generate(target, MTPDrafter(target, assistant), prompt, 20, k=3)
    assert out == expected


def test_drafter_needs_client_side_source_layers(tiny_checkpoint, assistant):
    if variant(tiny_checkpoint) == "ple":
        pytest.skip("the ple layout always keeps the source layers on the client")
    with ThreadedSwarm(tiny_checkpoint, [(1, 8)], model="tiny") as swarm:
        target = RemotePipeline.connect(str(tiny_checkpoint), swarm.tracker_url, model="tiny", first_layers=1,
                                        last_layers=0, dtype=torch.float32)
        with target:
            prompt = tokens(4)
            first = int(target.forward(prompt, 0)[-1].argmax())
            with pytest.raises(KeyError, match="runs on a peer"):
                MTPDrafter(target, assistant).propose(prompt + [first], 2, GREEDY)
