import torch

from adaptive_input import make_keep_mask
from model import GPT, GPTConfig


def test_dense_policy_preserves_every_input():
    tokens = torch.tensor([[1, 2, 3, 4]])
    assert make_keep_mask(tokens).all()


def test_periodic_policy_has_expected_phase():
    tokens = torch.zeros((2, 8), dtype=torch.long)
    expected = torch.tensor([True, False, True, False, True, False, True, False])
    assert torch.equal(make_keep_mask(tokens, "periodic", 0.5)[0], expected)


def test_model_accepts_a_causal_input_budget_mask():
    model = GPT(GPTConfig(block_size=8, vocab_size=16, n_layer=1, n_head=1, n_embd=8, adaptive_input=True))
    x = torch.randint(0, 16, (2, 8))
    logits, loss = model(x, x, make_keep_mask(x, "periodic", 0.5))
    assert logits.shape == (2, 8, 16)
    assert loss is not None
