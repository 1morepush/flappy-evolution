"""
Pydantic models used on the WebSocket and REST API wire.
All lists are kept compact (parallel arrays) for minimal JSON size.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Game state (pushed to clients)
# ---------------------------------------------------------------------------

class BirdsArray(BaseModel):
    """Parallel arrays for all P birds — much smaller than P BirdState objects."""
    ys: list[float]           # y positions
    vys: list[float]          # y velocities
    alive: list[bool]


class PipeState(BaseModel):
    x: float
    gap_top: float            # y coordinate of top pipe's bottom edge
    gap_bottom: float         # y coordinate of bottom pipe's top edge


class StateFrame(BaseModel):
    """Full snapshot sent on every tick in WATCH mode (sampled in TURBO)."""
    type: Literal["state"] = "state"
    tick: int
    birds: BirdsArray
    pipes: list[PipeState]
    # Per-layer activations for the focused bird: [[input], [hidden], [output]]
    focus_activations: list[list[float]]
    focus_id: int             # index of the focused bird


class StatsFrame(BaseModel):
    """Lightweight stats pushed frequently in both modes."""
    type: Literal["stats"] = "stats"
    generation: int
    best_fitness: float
    avg_fitness: float
    population_size: int
    birds_alive: int
    best_fitness_ever: float
    mode: str                 # "WATCH" | "TURBO" | "REPLAY" | "PAUSED"
    ticks_this_gen: int


class GenerationRecord(BaseModel):
    """Emitted once per generation rollover — appended to analytics history."""
    type: Literal["generation"] = "generation"
    generation: int
    best_fitness: float
    avg_fitness: float
    max_survival_ticks: int
    improvement_rate: float   # (best - prev_best) / max(prev_best, 1)
    pipe_seed: int            # RNG seed used — enables deterministic replay


# ---------------------------------------------------------------------------
# REST response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    generation: int
    mode: str


class TopologyResponse(BaseModel):
    layers: list[int]          # e.g. [5, 8, 2]
    input_labels: list[str]
    output_labels: list[str]


class AnalyticsResponse(BaseModel):
    history: list[GenerationRecord]


class ReplaySummary(BaseModel):
    generation: int
    best_fitness: float
    pipe_seed: int


class ReplayListResponse(BaseModel):
    replays: list[ReplaySummary]


class ReplayDetail(BaseModel):
    generation: int
    best_fitness: float
    pipe_seed: int
    genome: list[float]        # flat float32 gene vector
    topology: list[int]


# ---------------------------------------------------------------------------
# Client → Server control messages (received over WebSocket)
# ---------------------------------------------------------------------------

class SetModeMsg(BaseModel):
    type: Literal["set_mode"]
    mode: str                  # "WATCH" | "TURBO"


class PauseMsg(BaseModel):
    type: Literal["pause"]


class ResumeMsg(BaseModel):
    type: Literal["resume"]


class ResetMsg(BaseModel):
    type: Literal["reset"]


class SetSpeedMsg(BaseModel):
    type: Literal["set_speed"]
    turbo_steps: int           # steps per loop in TURBO


class FocusBirdMsg(BaseModel):
    type: Literal["focus_bird"]
    bird_id: int


class SetMutationMsg(BaseModel):
    type: Literal["set_mutation"]
    rate: float
    sigma: float


class LoadReplayMsg(BaseModel):
    type: Literal["load_replay"]
    generation: int
