"""
Selection strategies for the genetic algorithm.
"""
from __future__ import annotations

import numpy as np


def elitism_indices(fitnesses: np.ndarray, elite_count: int) -> list[int]:
    """Return indices of the top `elite_count` genomes by fitness."""
    return list(np.argsort(fitnesses)[::-1][:elite_count])


def tournament_select(
    fitnesses: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> int:
    """
    Tournament selection: pick `tournament_size` candidates at random and
    return the index of the one with the highest fitness.
    """
    n = len(fitnesses)
    candidates = rng.integers(0, n, size=tournament_size)
    best = candidates[int(np.argmax(fitnesses[candidates]))]
    return int(best)
