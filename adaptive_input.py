"""Causal input-budget policies used by the first adaptive-input experiments.

This module intentionally separates *which observations are retained* from the
language model.  That makes the dense model, periodic subsampling, and a later
learned controller comparable under the exact same architecture and loss.
"""

import torch


def make_keep_mask(tokens, policy="dense", keep_rate=1.0):
    """Return a deterministic causal keep mask for a (batch, time) token tensor.

    `periodic` retains the first token and then every n-th observation. It is
    deliberately content-independent: it is the rate-distortion control before
    claiming that a controller learned anything useful from the content.
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, time)")
    if policy == "dense":
        return torch.ones_like(tokens, dtype=torch.bool)
    if policy != "periodic":
        raise ValueError(f"unknown input policy: {policy}")
    if not 0.0 < keep_rate <= 1.0:
        raise ValueError("keep_rate must be in (0, 1]")

    stride = max(1, round(1.0 / keep_rate))
    positions = torch.arange(tokens.size(1), device=tokens.device)
    return (positions % stride == 0).unsqueeze(0).expand_as(tokens)
