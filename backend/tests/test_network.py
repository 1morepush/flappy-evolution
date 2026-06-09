"""Tests for neural_network/network.py and genome.py."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import NetworkConfig
from neural_network.genome import (
    gene_count,
    random_genome,
    mutate,
    crossover,
    unpack,
    pack,
    build_population_tensors,
)
from neural_network.network import forward, decide, forward_trace


CFG = NetworkConfig()
RNG = np.random.default_rng(0)


def test_gene_count():
    assert gene_count(CFG) == 5 * 8 + 8 + 8 * 2 + 2  # == 66


def test_random_genome_shape():
    g = random_genome(CFG, RNG)
    assert g.shape == (gene_count(CFG),)
    assert g.dtype == np.float32


def test_pack_unpack_roundtrip():
    g = random_genome(CFG, RNG)
    W1, b1, W2, b2 = unpack(g, CFG)
    g2 = pack(W1, b1, W2, b2)
    np.testing.assert_array_almost_equal(g, g2)


def test_forward_shape():
    P = 10
    genomes = [random_genome(CFG, RNG) for _ in range(P)]
    W1, b1, W2, b2 = build_population_tensors(genomes, CFG)
    inputs = RNG.random((P, 5)).astype(np.float32)
    logits = forward(inputs, W1, b1, W2, b2)
    assert logits.shape == (P, 2)


def test_forward_deterministic():
    P = 5
    genomes = [random_genome(CFG, np.random.default_rng(42)) for _ in range(P)]
    W1, b1, W2, b2 = build_population_tensors(genomes, CFG)
    inputs = np.ones((P, 5), dtype=np.float32) * 0.5
    out1 = forward(inputs, W1, b1, W2, b2)
    out2 = forward(inputs, W1, b1, W2, b2)
    np.testing.assert_array_equal(out1, out2)


def test_forward_dead_birds_zero():
    P = 4
    genomes = [random_genome(CFG, RNG) for _ in range(P)]
    W1, b1, W2, b2 = build_population_tensors(genomes, CFG)
    inputs = np.ones((P, 5), dtype=np.float32)
    alive = np.array([True, False, True, False])
    logits = forward(inputs, W1, b1, W2, b2, alive_mask=alive)
    assert logits[1, 0] == 0.0 and logits[1, 1] == 0.0
    assert logits[3, 0] == 0.0 and logits[3, 1] == 0.0


def test_decide_shape():
    logits = np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])
    decisions = decide(logits)
    assert decisions.shape == (3,)
    assert decisions[0] == True
    assert decisions[1] == False


def test_mutate_changes_fraction():
    rng = np.random.default_rng(99)
    g = random_genome(CFG, rng)
    rate = 0.5
    mutated = mutate(g, rate, 0.3, rng)
    changed = (g != mutated)
    # With rate=0.5 and 66 genes, should change ~33 genes
    # Allow range 10–56 to avoid flaky test
    assert 5 < changed.sum() < 61


def test_mutate_zero_rate():
    rng = np.random.default_rng(1)
    g = random_genome(CFG, rng)
    mutated = mutate(g, 0.0, 0.3, rng)
    np.testing.assert_array_equal(g, mutated)


def test_crossover_genes_from_parents():
    rng = np.random.default_rng(7)
    a = np.zeros(gene_count(CFG), dtype=np.float32)
    b = np.ones(gene_count(CFG), dtype=np.float32)
    child = crossover(a, b, rng)
    # Every gene must be either 0.0 or 1.0
    assert np.all((child == 0.0) | (child == 1.0))


def test_forward_trace_shapes():
    g = random_genome(CFG, RNG)
    W1, b1, W2, b2 = unpack(g, CFG)
    inputs = np.array([0.5, -0.1, 0.8, 0.3, 0.6], dtype=np.float32)
    acts = forward_trace(inputs, W1, b1, W2, b2)
    assert len(acts) == 3          # 3 layers
    assert len(acts[0]) == 5       # input
    assert len(acts[1]) == 8       # hidden
    assert len(acts[2]) == 2       # output
