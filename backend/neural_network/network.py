"""
Vectorized feedforward neural network for the full population.

Architecture: 5 → 8 → 2
  - Hidden activation: tanh
  - Output: raw logits (argmax → 0=nothing, 1=flap)

All P birds are processed in a single batched matrix multiply so there are
no Python per-bird loops in the hot path.

Weight tensors live externally (in Population) and are passed in so this
module stays stateless and easy to test.
"""
from __future__ import annotations

import numpy as np
from app.config import NetworkConfig


def forward(
    inputs: np.ndarray,           # (P, 5)  float32
    W1: np.ndarray,               # (P, 5, 8)
    b1: np.ndarray,               # (P, 8)
    W2: np.ndarray,               # (P, 8, 2)
    b2: np.ndarray,               # (P, 2)
    alive_mask: np.ndarray | None = None,  # (P,) bool — skip dead birds
) -> np.ndarray:
    """
    Batched forward pass.  Returns (P, 2) logits.
    Dead birds (alive_mask=False) get zero logits.
    """
    P = inputs.shape[0]

    if alive_mask is None:
        alive_mask = np.ones(P, dtype=bool)

    logits = np.zeros((P, 2), dtype=np.float32)

    if not alive_mask.any():
        return logits

    x = inputs[alive_mask]            # (A, 5)
    w1 = W1[alive_mask]               # (A, 5, 8)
    _b1 = b1[alive_mask]              # (A, 8)
    w2 = W2[alive_mask]               # (A, 8, 2)
    _b2 = b2[alive_mask]              # (A, 2)

    # Hidden layer: (A, 5) @ (A, 5, 8)  →  (A, 8)
    h = np.einsum("ai,aij->aj", x, w1) + _b1  # (A, 8)
    h = np.tanh(h)

    # Output layer: (A, 8) @ (A, 8, 2)  →  (A, 2)
    out = np.einsum("ai,aij->aj", h, w2) + _b2  # (A, 2)

    logits[alive_mask] = out
    return logits


def decide(logits: np.ndarray) -> np.ndarray:
    """
    Convert (P, 2) logits to (P,) bool flap decisions.
    Flap when output[1] > output[0].
    """
    return logits[:, 1] > logits[:, 0]


def forward_trace(
    inputs_1d: np.ndarray,   # (5,) single bird
    W1_1d: np.ndarray,       # (5, 8)
    b1_1d: np.ndarray,       # (8,)
    W2_1d: np.ndarray,       # (8, 2)
    b2_1d: np.ndarray,       # (2,)
) -> list[list[float]]:
    """
    Single-bird forward pass that returns per-layer activations for the
    neural-network visualizer.
    Returns [[input_activations], [hidden_activations], [output_activations]].
    """
    x = inputs_1d.astype(np.float32)             # (5,)
    h = np.tanh(W1_1d.T @ x + b1_1d)            # (8,)
    out = W2_1d.T @ h + b2_1d                    # (2,)

    return [
        x.tolist(),
        h.tolist(),
        out.tolist(),
    ]


def get_topology(cfg: NetworkConfig) -> list[int]:
    return cfg.topology
