"""
Evolver: produces the next generation from the current population + fitnesses.

Pipeline per generation:
  1. Elitism  — copy the top E genomes unchanged
  2. Selection — tournament selection for parent pairs
  3. Crossover — uniform crossover
  4. Mutation  — Gaussian noise on a fraction of genes
"""
from __future__ import annotations

import numpy as np
from app.config import EvolutionConfig, NetworkConfig
from neural_network.genome import mutate, crossover
from evolution.selection import elitism_indices, tournament_select


class Evolver:
    def __init__(self, net_cfg: NetworkConfig, evo_cfg: EvolutionConfig, seed: int = 42) -> None:
        self.net_cfg = net_cfg
        self.evo_cfg = evo_cfg
        self.rng = np.random.default_rng(seed)

    def next_generation(
        self,
        genomes: list[np.ndarray],
        fitnesses: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Return a new list of genomes of the same length.
        Uses the configured elite_count, tournament_size, mutation_rate, sigma.
        """
        cfg = self.evo_cfg
        n = len(genomes)
        new_gen: list[np.ndarray] = []

        # 1. Elitism
        elite_idx = elitism_indices(fitnesses, cfg.elite_count)
        for idx in elite_idx:
            new_gen.append(genomes[idx].copy())

        # 2. Fill the rest via tournament → crossover → mutation
        while len(new_gen) < n:
            p1_idx = tournament_select(fitnesses, cfg.tournament_size, self.rng)
            p2_idx = tournament_select(fitnesses, cfg.tournament_size, self.rng)
            child = crossover(genomes[p1_idx], genomes[p2_idx], self.rng)
            child = mutate(child, cfg.mutation_rate, cfg.mutation_sigma, self.rng)
            new_gen.append(child)

        return new_gen[:n]

    def update_mutation(self, rate: float, sigma: float) -> None:
        self.evo_cfg.mutation_rate = rate
        self.evo_cfg.mutation_sigma = sigma
