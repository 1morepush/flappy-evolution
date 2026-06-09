"""Tests for evolution/ modules."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import NetworkConfig, EvolutionConfig
from neural_network.genome import random_genome, gene_count
from evolution.selection import elitism_indices, tournament_select
from evolution.evolver import Evolver
from evolution.population import Population

NET_CFG = NetworkConfig()
EVO_CFG = EvolutionConfig(population_size=20, elite_count=2, tournament_size=3)
RNG = np.random.default_rng(0)


def make_genomes(n=20):
    rng = np.random.default_rng(42)
    return [random_genome(NET_CFG, rng) for _ in range(n)]


def test_elitism_indices_top():
    fitnesses = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
    top2 = elitism_indices(fitnesses, 2)
    assert 3 in top2  # fitness 9.0 is best
    assert 1 in top2  # fitness 5.0 is second


def test_tournament_select_returns_valid_index():
    fitnesses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = tournament_select(fitnesses, 3, rng)
        assert 0 <= idx < len(fitnesses)


def test_elitism_preserved():
    """Elite genomes must appear unchanged in the next generation."""
    genomes = make_genomes(20)
    fitnesses = np.arange(20, dtype=np.float32)  # bird 19 is best
    evolver = Evolver(NET_CFG, EVO_CFG, seed=0)
    new_genomes = evolver.next_generation(genomes, fitnesses)
    assert len(new_genomes) == 20
    # Top 2 should match exactly
    elite_idx = elitism_indices(fitnesses, 2)
    for idx in elite_idx:
        found = any(np.array_equal(genomes[idx], g) for g in new_genomes)
        assert found, f"Elite genome {idx} not found in next generation"


def test_next_generation_size():
    genomes = make_genomes(20)
    fitnesses = np.random.default_rng(0).random(20).astype(np.float32)
    evolver = Evolver(NET_CFG, EVO_CFG, seed=1)
    new_gen = evolver.next_generation(genomes, fitnesses)
    assert len(new_gen) == 20


def test_population_replace():
    pop = Population(NET_CFG, EVO_CFG, seed=0)
    old_genomes = [g.copy() for g in pop.genomes]
    new_genomes = make_genomes(EVO_CFG.population_size)
    pop.replace_genomes(new_genomes)
    # Tensors should be rebuilt to new shapes
    assert pop.W1.shape[0] == EVO_CFG.population_size


def test_population_best_fitness():
    pop = Population(NET_CFG, EVO_CFG, seed=0)
    fitnesses = np.zeros(EVO_CFG.population_size, dtype=np.float32)
    fitnesses[5] = 99.0
    pop.record_fitnesses(fitnesses)
    assert pop.best_fitness() == pytest.approx(99.0)
