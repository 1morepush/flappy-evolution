"""
Vectorized game world — manages all P birds and the pipe course simultaneously.

Key design:
- Bird state is stored as parallel NumPy arrays (ys, vys, alive).
- Pipes are a ring-buffer of (x, gap_top, gap_bottom) rows.
- Forward pass (step) is one NumPy call chain with no per-bird Python loops.
- A per-generation RNG seed makes every generation's pipe layout reproducible
  for fair fitness comparison and deterministic replay.
"""
from __future__ import annotations

import numpy as np
from app.config import GameConfig, EvolutionConfig
from game.physics import (
    apply_gravity,
    apply_flap,
    generate_pipes,
    check_collisions,
)

# How many pipes to keep in the buffer (enough that one is always visible)
N_PIPES = 6


class World:
    """Mutable simulation state for one generation."""

    def __init__(
        self,
        population_size: int,
        game_cfg: GameConfig,
        evo_cfg: EvolutionConfig,
        pipe_seed: int,
    ) -> None:
        self.P = population_size
        self.cfg = game_cfg
        self.evo_cfg = evo_cfg
        self.pipe_seed = pipe_seed
        self.tick = 0

        # Bird state — shape (P,)
        start_y = float(game_cfg.height // 2)
        self.ys: np.ndarray = np.full(self.P, start_y, dtype=np.float32)
        self.vys: np.ndarray = np.zeros(self.P, dtype=np.float32)
        self.alive: np.ndarray = np.ones(self.P, dtype=bool)

        # Fitness accumulators — shape (P,)
        self.fitness: np.ndarray = np.zeros(self.P, dtype=np.float32)

        # Pipe ring-buffer — shape (N_PIPES, 3): [x, gap_top, gap_bottom]
        rng = np.random.default_rng(pipe_seed)
        self.pipes: np.ndarray = generate_pipes(N_PIPES, game_cfg, rng)
        self._rng = rng  # keep for recycling pipes

        # Track which pipes each bird has passed (for centre bonus)
        self._pipes_passed = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def birds_alive(self) -> int:
        return int(self.alive.sum())

    @property
    def all_dead(self) -> bool:
        return not self.alive.any()

    def get_sensors(self) -> np.ndarray:
        """
        Return normalized sensor inputs for all P birds: shape (P, 5).
        Inputs:
          0 – bird Y  (0=top, 1=bottom of play area)
          1 – bird vertical velocity  (normalised to ±1 range)
          2 – horizontal distance to next pipe  (0..1)
          3 – next pipe gap_top  (0..1)
          4 – next pipe gap_bottom  (0..1)
        """
        h = float(self.cfg.height)
        w = float(self.cfg.width)
        gnd = float(self.cfg.ground_y)
        bx = float(self.cfg.bird_x)

        # Find the "next" pipe (first pipe whose right edge is ahead of the bird)
        pipe_rights = self.pipes[:, 0] + self.cfg.pipe_width / 2
        ahead = np.where(pipe_rights > bx - self.cfg.bird_radius)[0]
        if len(ahead) == 0:
            # Fallback: just take the first pipe (shouldn't happen in practice)
            next_idx = 0
        else:
            next_idx = int(ahead[0])

        nx, gap_top, gap_bottom = self.pipes[next_idx]

        dist = float(nx - bx)

        # Normalise
        y_norm = self.ys / gnd                             # (P,)
        vy_norm = np.clip(self.vys / 15.0, -1.0, 1.0)     # (P,)
        dist_norm = np.clip(dist / w, 0.0, 1.0)            # scalar
        top_norm = float(gap_top / gnd)
        bot_norm = float(gap_bottom / gnd)

        sensors = np.column_stack([
            y_norm,
            vy_norm,
            np.full(self.P, dist_norm, dtype=np.float32),
            np.full(self.P, top_norm, dtype=np.float32),
            np.full(self.P, bot_norm, dtype=np.float32),
        ]).astype(np.float32)

        return sensors

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self, flap_decisions: np.ndarray) -> None:
        """
        Advance the world by one tick.
        flap_decisions: (P,) bool — True = flap this tick.
        Only live birds are actually updated; dead birds remain frozen.
        """
        if self.all_dead:
            return

        live = self.alive

        # Physics (operate only on live birds to save ops, but keep shapes)
        new_vy = apply_gravity(self.vys, self.cfg)
        new_vy = apply_flap(new_vy, flap_decisions & live, self.cfg)
        new_y = self.ys + new_vy

        # Apply only to live birds
        self.vys = np.where(live, new_vy, self.vys)
        self.ys = np.where(live, new_y, self.ys)

        # Pipe movement
        self.pipes[:, 0] -= self.cfg.pipe_speed

        # Recycle pipes that have scrolled off the left
        for i in range(N_PIPES):
            if self.pipes[i, 0] < -self.cfg.pipe_width:
                # Place this pipe at the rightmost position + spacing
                right_x = self.pipes[:, 0].max() + self.cfg.pipe_spacing
                self.pipes[i, 0] = right_x
                new_pipe = generate_pipes(1, self.cfg, self._rng)
                self.pipes[i, 1] = new_pipe[0, 1]   # gap_top
                self.pipes[i, 2] = new_pipe[0, 2]   # gap_bottom

        # Collision detection — returns (P,) bool (True = still alive)
        still_alive = check_collisions(self.ys, float(self.cfg.bird_x), self.pipes, self.cfg)
        self.alive = self.alive & still_alive

        # Fitness: +1 per tick survived (only live birds)
        self.fitness += self.alive.astype(np.float32)

        # Pipe-passing bonus for live birds
        bx = float(self.cfg.bird_x)
        pipe_centres = self.pipes[:, 0]
        for i, cx in enumerate(pipe_centres):
            if (bx - self.cfg.pipe_speed) <= cx < bx:
                # All currently-alive birds just passed this pipe centre
                gap_mid = (self.pipes[i, 1] + self.pipes[i, 2]) / 2.0
                # Reward alive birds near the centre of the gap
                if self.alive.any():
                    self.fitness[self.alive] += self.evo_cfg.pipe_reward

        self.tick += 1
