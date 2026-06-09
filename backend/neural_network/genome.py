"""
Genome encoding: a flat float32 vector of neural-network weights.

Layout for 5→8→2:
  W1  (5*8 = 40)  + b1 (8)  +  W2 (8*2 = 16)  + b2 (2)  = 66 genes

Operations:
  - pack/unpack between flat vector and layer matrices
  - random initialisation (Xavier)
  - mutation (Gaussian noise on a random subset of genes)
  - crossover (uniform)
"""
from __future__ import annotations

import numpy as np
from app.config import NetworkConfig


def gene_count(cfg: NetworkConfig) -> int:
    i, h, o = cfg.input_size, cfg.hidden_size, cfg.output_size
    return i * h + h + h * o + o


def unpack(genes: np.ndarray, cfg: NetworkConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flat vector → (W1, b1, W2, b2).
    W1: (input, hidden), b1: (hidden,), W2: (hidden, output), b2: (output,)
    """
    i, h, o = cfg.input_size, cfg.hidden_size, cfg.output_size
    idx = 0
    W1 = genes[idx: idx + i * h].reshape(i, h);  idx += i * h
    b1 = genes[idx: idx + h];                     idx += h
    W2 = genes[idx: idx + h * o].reshape(h, o);   idx += h * o
    b2 = genes[idx: idx + o]
    return W1, b1, W2, b2


def pack(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Layer matrices → flat float32 gene vector."""
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()]).astype(np.float32)


def random_genome(cfg: NetworkConfig, rng: np.random.Generator) -> np.ndarray:
    """Xavier-initialised random genome."""
    i, h, o = cfg.input_size, cfg.hidden_size, cfg.output_size
    W1 = rng.standard_normal((i, h)).astype(np.float32) * np.sqrt(2.0 / i)
    b1 = np.zeros(h, dtype=np.float32)
    W2 = rng.standard_normal((h, o)).astype(np.float32) * np.sqrt(2.0 / h)
    b2 = np.zeros(o, dtype=np.float32)
    return pack(W1, b1, W2, b2)


def mutate(
    genome: np.ndarray,
    rate: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a mutated copy of genome.
    Each gene is independently perturbed with probability `rate`
    by Gaussian noise N(0, sigma).
    """
    child = genome.copy()
    mask = rng.random(len(child)) < rate
    if mask.any():
        child[mask] += rng.standard_normal(mask.sum()).astype(np.float32) * sigma
    return child


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Uniform crossover: each gene is drawn from parent_a or parent_b
    with equal probability.
    """
    mask = rng.random(len(parent_a)) < 0.5
    child = np.where(mask, parent_a, parent_b).astype(np.float32)
    return child


# ------------------------------------------------------------------
# Population tensor helpers
# ------------------------------------------------------------------

def build_population_tensors(
    genomes: list[np.ndarray],
    cfg: NetworkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack a list of P genomes into stacked weight tensors ready for
    the batched forward pass.
    Returns:
      W1: (P, input, hidden)
      b1: (P, hidden)
      W2: (P, hidden, output)
      b2: (P, output)
    """
    W1s, b1s, W2s, b2s = [], [], [], []
    for g in genomes:
        W1, b1, W2, b2 = unpack(g, cfg)
        W1s.append(W1)
        b1s.append(b1)
        W2s.append(W2)
        b2s.append(b2)
    return (
        np.stack(W1s, axis=0),   # (P, i, h)
        np.stack(b1s, axis=0),   # (P, h)
        np.stack(W2s, axis=0),   # (P, h, o)
        np.stack(b2s, axis=0),   # (P, o)
    )
