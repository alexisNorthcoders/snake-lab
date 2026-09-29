"""Exporting a policy as a brain file: the JSON format of `snake-colyseus/bots`, which any part of the game plays."""

import numpy as np
import torch
from torch import nn

BRAIN_FORMAT = "snake-brain"
BRAIN_FORMAT_VERSION = 1
# The engine's RULES_VERSION the lab is pinned to; tests check it against the package.
RULES_VERSION = 2


def export_brain(policy: nn.Sequential, encoder: int) -> dict:
    """The policy's Linear layers as a brain: `tanh` hidden layers, and a linear output (the three logits).

    A brain's `weights` are output × input, which is how `nn.Linear` stores them, so they go across as they are.
    The brain decider takes the highest output, which is the policy's most likely move.
    """
    linears = [layer for layer in policy if isinstance(layer, nn.Linear)]
    layers = []
    for linear in linears:
        weights = linear.weight.detach().to(torch.float64).numpy()
        biases = linear.bias.detach().to(torch.float64).numpy()
        if not (np.isfinite(weights).all() and np.isfinite(biases).all()):
            raise ValueError("the policy has non-finite weights: it can't be exported")
        layers.append({"weights": weights.tolist(), "biases": biases.tolist()})
    return {
        "format": BRAIN_FORMAT,
        "formatVersion": BRAIN_FORMAT_VERSION,
        "encoderVersion": encoder,
        "rulesVersion": RULES_VERSION,
        "sizes": [linears[0].in_features] + [linear.out_features for linear in linears],
        "activation": "tanh",
        "layers": layers,
    }
