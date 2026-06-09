"""
Population: holds the list of genomes and their current fitnesses.
Keeps weight tensors in sync for the vectorized forward pass.
"""
from __future__ import annotations

import numpy as np
from app.config import NetworkConfig, EvolutionConfig
from neural_network.genome import random_genome, build_population_tensors


class Population:
    def __init__(self, net_cfg: NetworkConfig, evo_cfg: EvolutionConfig, seed: int = 0) -> None:
        self.net_cfg = net_cfg
        self.evo_cfg = evo_cfg
        self.rng = np.random.default_rng(seed)
        self.size = evo_cfg.population_size

        # List of flat float32 gene vectors, one per bird
        self.genomes: list[np.ndarray] = [
            random_genome(net_cfg, self.rng) for _ in range(self.size)
        ]

        # Stacked tensors for batched forward pass — rebuilt after each generation
        self._rebuild_tensors()

        # Fitness values assigned at the end of each generation
        self.fitnesses: np.ndarray = np.zeros(self.size, dtype=np.float32)

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def _rebuild_tensors(self) -> None:
        self.W1, self.b1, self.W2, self.b2 = build_population_tensors(
            self.genomes, self.net_cfg
        )

    # ------------------------------------------------------------------
    # After a generation completes
    # ------------------------------------------------------------------

    def record_fitnesses(self, fitnesses: np.ndarray) -> None:
        """Store fitnesses from the world at end of generation."""
        self.fitnesses = fitnesses.copy()

    def best_genome(self) -> np.ndarray:
        return self.genomes[int(np.argmax(self.fitnesses))]

    def best_fitness(self) -> float:
        return float(self.fitnesses.max()) if len(self.fitnesses) else 0.0

    def avg_fitness(self) -> float:
        return float(self.fitnesses.mean()) if len(self.fitnesses) else 0.0

    def replace_genomes(self, new_genomes: list[np.ndarray]) -> None:
        """Called by evolver after producing the next generation."""
        self.genomes = new_genomes
        self.fitnesses = np.zeros(self.size, dtype=np.float32)
        self._rebuild_tensors()
