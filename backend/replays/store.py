"""
Replay store: persists and loads the best genome from each generation
as compact JSON files on disk.

File naming: replays_data/gen_{generation:05d}.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from app.schemas import ReplayDetail, ReplaySummary

_DATA_DIR = Path(__file__).parent.parent / "replays_data"


def _gen_path(generation: int) -> Path:
    _DATA_DIR.mkdir(exist_ok=True)
    return _DATA_DIR / f"gen_{generation:05d}.json"


def save(
    generation: int,
    best_fitness: float,
    pipe_seed: int,
    genome: np.ndarray,
    topology: list[int],
) -> None:
    data = {
        "generation": generation,
        "best_fitness": float(best_fitness),
        "pipe_seed": int(pipe_seed),
        "genome": genome.astype(np.float32).tolist(),
        "topology": topology,
    }
    with open(_gen_path(generation), "w") as f:
        json.dump(data, f)


def load(generation: int) -> ReplayDetail | None:
    path = _gen_path(generation)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return ReplayDetail(**data)


def list_replays() -> list[ReplaySummary]:
    if not _DATA_DIR.exists():
        return []
    summaries = []
    for path in sorted(_DATA_DIR.glob("gen_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            summaries.append(
                ReplaySummary(
                    generation=data["generation"],
                    best_fitness=data["best_fitness"],
                    pipe_seed=data["pipe_seed"],
                )
            )
        except Exception:
            continue
    return summaries


def clear() -> None:
    """Remove all saved replays (used on full reset)."""
    if not _DATA_DIR.exists():
        return
    for path in _DATA_DIR.glob("gen_*.json"):
        path.unlink(missing_ok=True)
