"""Tests for game/physics.py and game/world.py."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import GameConfig, EvolutionConfig
from game.physics import check_collisions, generate_pipes
from game.world import World

GAME_CFG = GameConfig()
EVO_CFG = EvolutionConfig(population_size=10)


def make_world(seed=0):
    return World(
        population_size=10,
        game_cfg=GAME_CFG,
        evo_cfg=EVO_CFG,
        pipe_seed=seed,
    )


def test_world_initial_state():
    w = make_world()
    assert w.ys.shape == (10,)
    assert w.alive.all()
    assert w.tick == 0


def test_world_step_advances_tick():
    w = make_world()
    flap = np.zeros(10, dtype=bool)
    w.step(flap)
    assert w.tick == 1


def test_gravity_decreases_y():
    """With no flap, birds fall (vy increases, y increases towards ground)."""
    w = make_world()
    init_vys = w.vys.copy()
    w.step(np.zeros(10, dtype=bool))
    assert (w.vys > init_vys).all()  # vy increased (gravity added)


def test_flap_impulse():
    w = make_world()
    flap = np.ones(10, dtype=bool)
    w.step(flap)
    # After flap, vy should be the flap_impulse (negative = upward)
    assert (w.vys < 0).all()


def test_ground_collision_kills_bird():
    """Bird at y > ground_y should die on next step."""
    w = make_world()
    # Force a bird below the ground
    w.ys[3] = GAME_CFG.ground_y + 5
    w.step(np.zeros(10, dtype=bool))
    assert not w.alive[3]


def test_ceiling_collision_kills_bird():
    w = make_world()
    w.ys[0] = -5   # above ceiling
    w.step(np.zeros(10, dtype=bool))
    assert not w.alive[0]


def test_pipe_collision_kills_bird():
    """Bird overlapping a pipe body should die."""
    w = make_world()
    # Place a pipe exactly at the bird's x position
    bx = float(GAME_CFG.bird_x)
    # Set pipe x == bird_x
    w.pipes[0, 0] = bx
    # Set gap so the bird is NOT in the gap (bird in the solid part)
    w.pipes[0, 1] = GAME_CFG.height * 0.6  # gap_top
    w.pipes[0, 2] = GAME_CFG.height * 0.9  # gap_bottom
    w.ys[0] = GAME_CFG.height * 0.3        # bird above the gap → in top pipe solid
    w.step(np.zeros(10, dtype=bool))
    assert not w.alive[0]


def test_sensors_shape():
    w = make_world()
    sensors = w.get_sensors()
    assert sensors.shape == (10, 5)
    # All sensor values should be in a reasonable range
    assert np.all(sensors >= -2.0) and np.all(sensors <= 2.0)


def test_generate_pipes_shape():
    rng = np.random.default_rng(0)
    pipes = generate_pipes(5, GAME_CFG, rng)
    assert pipes.shape == (5, 3)
    # gap_bottom - gap_top should equal pipe_gap for all pipes
    gaps = pipes[:, 2] - pipes[:, 1]
    np.testing.assert_allclose(gaps, GAME_CFG.pipe_gap, atol=1)
