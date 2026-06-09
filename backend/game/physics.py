"""
Physics constants and helpers.
All actual simulation state lives in world.py; this module only defines
the rules of motion so they can be imported without side-effects.
"""
from __future__ import annotations
import numpy as np
from app.config import GameConfig


def apply_gravity(vy: np.ndarray, cfg: GameConfig) -> np.ndarray:
    """Return new vy after one gravity tick (vectorized over P birds)."""
    new_vy = vy + cfg.gravity
    # Terminal velocity cap so birds don't fall infinitely fast
    return np.clip(new_vy, -20.0, 15.0)


def apply_flap(vy: np.ndarray, flap_mask: np.ndarray, cfg: GameConfig) -> np.ndarray:
    """Apply flap impulse where flap_mask is True."""
    return np.where(flap_mask, cfg.flap_impulse, vy)


def next_pipe_x(current_x: float, cfg: GameConfig) -> float:
    """Wrap a pipe back to the right edge once it passes off-screen."""
    return current_x + cfg.width + cfg.pipe_spacing


def generate_pipes(n: int, cfg: GameConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Generate n pipe pairs.
    Returns shape (n, 3) float32: [x, gap_top, gap_bottom].

    gap_top  = y of the bottom edge of the top pipe
    gap_bottom = y of the top edge of the bottom pipe
    gap_bottom - gap_top == cfg.pipe_gap always.
    """
    margin = 60
    min_y = margin + cfg.pipe_gap // 2
    max_y = cfg.ground_y - margin - cfg.pipe_gap // 2

    centres = rng.integers(min_y, max_y + 1, size=n).astype(np.float32)
    gap_tops = centres - cfg.pipe_gap // 2
    gap_bottoms = centres + cfg.pipe_gap // 2

    xs = np.array([
        cfg.bird_x + cfg.pipe_spacing + i * cfg.pipe_spacing
        for i in range(n)
    ], dtype=np.float32)

    return np.stack([xs, gap_tops, gap_bottoms], axis=1)


def check_collisions(
    bird_y: np.ndarray,
    bird_x: float,
    pipes: np.ndarray,
    cfg: GameConfig,
) -> np.ndarray:
    """
    Return boolean mask of birds that are alive after checking all pipes and bounds.
    bird_y: (P,) float32
    pipes: (N, 3) float32 [x, gap_top, gap_bottom]
    Returns (P,) bool — True = still alive
    """
    P = bird_y.shape[0]

    # Ground / ceiling check (vectorized over P)
    in_bounds = (bird_y > 0) & (bird_y + cfg.bird_radius < cfg.ground_y)

    # Pipe collision: for each pipe whose x-range overlaps the bird's x-range
    r = cfg.bird_radius
    bx_left = bird_x - r
    bx_right = bird_x + r

    alive = in_bounds.copy()
    for px, gap_top, gap_bottom in pipes:
        pipe_left = px - cfg.pipe_width / 2
        pipe_right = px + cfg.pipe_width / 2

        x_overlap = (bx_right > pipe_left) & (bx_left < pipe_right)
        if not x_overlap:
            continue

        # Check vertical: bird must be inside the gap
        bird_top = bird_y - r
        bird_bot = bird_y + r
        in_gap = (bird_top > gap_top) & (bird_bot < gap_bottom)
        alive &= in_gap

    return alive
