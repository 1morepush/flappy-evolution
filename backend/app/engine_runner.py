"""
Engine runner: owns the authoritative simulation loop.

Runs as a persistent asyncio background task started at server startup.
All mutable state is synchronised by the asyncio event loop (single-threaded
by default) so no locking is needed.

Modes:
  WATCH  – one physics step per loop iteration, sleep to ~60fps
  TURBO  – N physics steps per iteration, no sleep; snapshots sampled by time
  REPLAY – single best-bird replay at watch speed, exits back to PAUSED
  PAUSED – loop suspends until mode changes
"""
from __future__ import annotations

import asyncio
import time
from typing import Literal

import numpy as np

from app.config import CONFIG, AppConfig
from app.schemas import GenerationRecord
from analytics.tracker import AnalyticsTracker
from evolution.evolver import Evolver
from evolution.population import Population
from game.world import World
from neural_network.genome import unpack
from neural_network.network import decide, forward, forward_trace
from replays.store import clear as clear_replays, load as load_replay, save as save_replay

SimMode = Literal["WATCH", "TURBO", "REPLAY", "PAUSED"]


class EngineRunner:
    def __init__(self, cfg: AppConfig = CONFIG) -> None:
        self.cfg = cfg
        self._clients: set[asyncio.Queue] = set()

        # Simulation state
        self.mode: SimMode = "WATCH"
        self.generation: int = 1
        self.best_fitness_ever: float = 0.0
        self._pipe_seed: int = 1

        self.population = Population(cfg.network, cfg.evolution)
        self.evolver = Evolver(cfg.network, cfg.evolution)
        self.world = self._new_world()
        self.tracker = AnalyticsTracker()

        # Focus bird (for NN visualizer)
        self.focus_id: int = 0

        # Replay sub-state
        self._replay_genome: np.ndarray | None = None
        self._replay_seed: int = 0
        self._replay_world: World | None = None

        # Control flags
        self._running = True
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Client connection management
    # ------------------------------------------------------------------

    def add_client(self, q: asyncio.Queue) -> None:
        self._clients.add(q)

    def remove_client(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    async def _broadcast(self, payload: dict) -> None:
        for q in list(self._clients):
            if q.full():
                try:
                    q.get_nowait()  # drop oldest frame to make room
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # still full after drop — skip this frame, client stays connected

    # ------------------------------------------------------------------
    # World factory
    # ------------------------------------------------------------------

    def _new_world(self) -> World:
        self._pipe_seed = int(np.random.default_rng().integers(0, 2**31))
        return World(
            population_size=self.cfg.evolution.population_size,
            game_cfg=self.cfg.game,
            evo_cfg=self.cfg.evolution,
            pipe_seed=self._pipe_seed,
        )

    # ------------------------------------------------------------------
    # One generation lifecycle
    # ------------------------------------------------------------------

    async def _finish_generation(self) -> None:
        """Score, evolve, record analytics, save replay, start new world."""
        fitnesses = self.world.fitness
        max_ticks = int(self.world.tick)

        best = float(fitnesses.max())
        avg = float(fitnesses.mean())
        if best > self.best_fitness_ever:
            self.best_fitness_ever = best

        self.population.record_fitnesses(fitnesses)

        # Analytics
        rec = self.tracker.record(
            generation=self.generation,
            best_fitness=best,
            avg_fitness=avg,
            max_survival_ticks=max_ticks,
            pipe_seed=self._pipe_seed,
        )
        await self._broadcast(rec.model_dump())

        # Save best genome replay
        save_replay(
            generation=self.generation,
            best_fitness=best,
            pipe_seed=self._pipe_seed,
            genome=self.population.best_genome(),
            topology=self.cfg.network.topology,
        )

        # Evolve
        new_genomes = self.evolver.next_generation(
            self.population.genomes, fitnesses
        )
        self.population.replace_genomes(new_genomes)

        self.generation += 1
        self.world = self._new_world()

        # Auto-focus on bird 0 (will be updated to best in first step)
        self.focus_id = 0

    # ------------------------------------------------------------------
    # Frame builders
    # ------------------------------------------------------------------

    def _build_state_frame(self, world: World, sensors: np.ndarray) -> dict:
        """Build state dict directly — avoids Pydantic model_dump() overhead."""
        alive_idx = np.where(world.alive)[0]
        if len(alive_idx) > 0:
            if world.alive[self.focus_id]:
                fid = self.focus_id
            else:
                masked = np.where(world.alive, world.fitness, -1.0)
                fid = int(np.argmax(masked))
            self.focus_id = fid
        else:
            fid = 0

        W1, b1, W2, b2 = unpack(self.population.genomes[fid], self.cfg.network)
        activations = forward_trace(sensors[fid], W1, b1, W2, b2)

        return {
            "type": "state",
            "tick": world.tick,
            "birds": {
                "ys": world.ys.tolist(),
                "vys": world.vys.tolist(),
                "alive": world.alive.tolist(),
            },
            "pipes": [
                {"x": float(p[0]), "gap_top": float(p[1]), "gap_bottom": float(p[2])}
                for p in world.pipes
            ],
            "focus_activations": activations,
            "focus_id": fid,
        }

    def _build_stats_frame(self, world: World) -> dict:
        return {
            "type": "stats",
            "generation": self.generation,
            "best_fitness": float(world.fitness.max()),
            "avg_fitness": float(world.fitness.mean()),
            "population_size": self.cfg.evolution.population_size,
            "birds_alive": world.birds_alive,
            "best_fitness_ever": self.best_fitness_ever,
            "mode": self.mode,
            "ticks_this_gen": world.tick,
        }

    # ------------------------------------------------------------------
    # Simulation step — returns sensors so caller can reuse them
    # ------------------------------------------------------------------

    def _sim_step(self, world: World) -> np.ndarray:
        """Advance one tick and return the sensors used (avoids recomputing)."""
        sensors = world.get_sensors()
        logits = forward(
            sensors,
            self.population.W1,
            self.population.b1,
            self.population.W2,
            self.population.b2,
            alive_mask=world.alive,
        )
        flap_decisions = decide(logits)
        world.step(flap_decisions)
        return sensors

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — runs until _running is False."""
        stats_counter = 0
        last_snapshot = time.monotonic()

        while self._running:
            if self.mode == "PAUSED":
                # Keep broadcasting stats so frontend knows we're paused
                stats = self._build_stats_frame(self.world)
                await self._broadcast(stats)
                await asyncio.sleep(0.1)
                continue

            if self.mode == "REPLAY":
                await self._replay_tick()
                continue

            # ---- WATCH or TURBO ----
            steps = 1 if self.mode == "WATCH" else self.cfg.turbo_steps_per_tick

            last_sensors: np.ndarray | None = None
            for _ in range(steps):
                if self.world.all_dead or self.world.tick >= self.cfg.game.max_ticks_per_gen:
                    await self._finish_generation()
                    stats_counter = 0
                    last_sensors = None
                    break
                last_sensors = self._sim_step(self.world)

            stats_counter += steps

            # Use sensors from the last step — no recomputation needed
            if last_sensors is None:
                last_sensors = self.world.get_sensors()

            # Broadcast state frame
            now = time.monotonic()
            if self.mode == "WATCH":
                frame = self._build_state_frame(self.world, last_sensors)
                await self._broadcast(frame)
            elif (now - last_snapshot) * 1000 >= self.cfg.turbo_snapshot_ms:
                frame = self._build_state_frame(self.world, last_sensors)
                await self._broadcast(frame)
                last_snapshot = now

            # Broadcast stats frame
            if stats_counter >= self.cfg.stats_every_ticks:
                stats = self._build_stats_frame(self.world)
                await self._broadcast(stats)
                stats_counter = 0

            # Sleep to pace WATCH mode at ~60fps
            if self.mode == "WATCH":
                await asyncio.sleep(1.0 / self.cfg.watch_fps)
            else:
                await asyncio.sleep(0)  # yield to event loop

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    async def _replay_tick(self) -> None:
        if self._replay_world is None or self._replay_genome is None:
            self.mode = "PAUSED"
            return

        rw = self._replay_world
        if rw.all_dead or rw.tick >= self.cfg.game.max_ticks_per_gen:
            self.mode = "PAUSED"
            self._replay_world = None
            return

        genome = self._replay_genome
        W1, b1, W2, b2 = unpack(genome, self.cfg.network)
        sensors = rw.get_sensors()

        W1b = W1[None]; b1b = b1[None]; W2b = W2[None]; b2b = b2[None]
        logits = forward(sensors[0:1], W1b, b1b, W2b, b2b, alive_mask=rw.alive[0:1])
        flap = decide(logits)

        full_flap = np.zeros(self.cfg.evolution.population_size, dtype=bool)
        full_flap[0] = flap[0]
        rw.step(full_flap)

        activations = forward_trace(sensors[0], W1, b1, W2, b2)

        frame = {
            "type": "state",
            "tick": rw.tick,
            "birds": {
                "ys": rw.ys.tolist(),
                "vys": rw.vys.tolist(),
                "alive": rw.alive.tolist(),
            },
            "pipes": [
                {"x": float(p[0]), "gap_top": float(p[1]), "gap_bottom": float(p[2])}
                for p in rw.pipes
            ],
            "focus_activations": activations,
            "focus_id": 0,
        }
        await self._broadcast(frame)

        stats = {
            "type": "stats",
            "generation": self.generation,
            "best_fitness": float(rw.fitness[0]),
            "avg_fitness": float(rw.fitness[0]),
            "population_size": 1,
            "birds_alive": int(rw.alive[0]),
            "best_fitness_ever": self.best_fitness_ever,
            "mode": "REPLAY",
            "ticks_this_gen": rw.tick,
        }
        await self._broadcast(stats)

        await asyncio.sleep(1.0 / self.cfg.watch_fps)

    def start_replay(self, generation: int) -> bool:
        detail = load_replay(generation)
        if detail is None:
            return False
        import numpy as np
        genome = np.array(detail.genome, dtype=np.float32)
        self._replay_genome = genome
        self._replay_seed = detail.pipe_seed

        # One-bird replay world: use all 100 slots but only bird 0 is live
        rw = World(
            population_size=self.cfg.evolution.population_size,
            game_cfg=self.cfg.game,
            evo_cfg=self.cfg.evolution,
            pipe_seed=self._replay_seed,
        )
        # Kill all birds except bird 0
        rw.alive[1:] = False
        self._replay_world = rw
        self.mode = "REPLAY"
        return True

    # ------------------------------------------------------------------
    # Control commands
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode in ("WATCH", "TURBO"):
            self.mode = mode  # type: ignore[assignment]

    def pause(self) -> None:
        self._prev_mode = self.mode
        self.mode = "PAUSED"

    def resume(self) -> None:
        self.mode = getattr(self, "_prev_mode", "WATCH")  # type: ignore[assignment]

    def reset(self) -> None:
        self.generation = 1
        self.best_fitness_ever = 0.0
        self.population = Population(self.cfg.network, self.cfg.evolution)
        self.evolver = Evolver(self.cfg.network, self.cfg.evolution)
        self.world = self._new_world()
        self.tracker.reset()
        clear_replays()
        self.mode = "WATCH"

    def set_speed(self, turbo_steps: int) -> None:
        self.cfg.turbo_steps_per_tick = max(1, turbo_steps)

    def set_focus(self, bird_id: int) -> None:
        if 0 <= bird_id < self.cfg.evolution.population_size:
            self.focus_id = bird_id

    def set_mutation(self, rate: float, sigma: float) -> None:
        self.evolver.update_mutation(rate, sigma)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._task = asyncio.ensure_future(self.run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
