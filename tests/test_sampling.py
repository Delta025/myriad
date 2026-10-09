import torch

from myriad.client.sampling import GREEDY, Sampling, pick, probabilities


def test_temperature_top_k_top_p():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    p = probabilities(logits, Sampling(temperature=1.0))
    torch.testing.assert_close(p, logits.softmax(-1))
    assert (probabilities(logits, Sampling(temperature=1.0, top_k=2))[0, 2:] == 0).all()
    # top-p 0.7: the first token has about 0.64 of the mass, so the second is needed, the rest are dropped
    top_p = probabilities(logits, Sampling(temperature=1.0, top_p=0.7))[0]
    assert (top_p[:2] > 0).all() and (top_p[2:] == 0).all()
    torch.testing.assert_close(top_p.sum(), torch.tensor(1.0))
    assert probabilities(logits, Sampling(temperature=0.01))[0, 0] > 0.999


def test_pick():
    logits = torch.tensor([0.0, 3.0, 1.0])
    assert pick(logits, GREEDY) == 1
    g = torch.Generator().manual_seed(0)
    draws = [pick(logits, Sampling(temperature=1.0), g) for _ in range(2000)]
    assert abs(draws.count(1) / 2000 - logits.softmax(-1)[1].item()) < 0.04
