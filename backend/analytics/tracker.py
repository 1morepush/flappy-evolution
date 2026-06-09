"""
Analytics tracker: accumulates per-generation statistics and
exposes them as a list of GenerationRecord objects.
"""
from __future__ import annotations

from app.schemas import GenerationRecord


class AnalyticsTracker:
    def __init__(self) -> None:
        self.history: list[GenerationRecord] = []
        self._prev_best: float = 0.0

    def record(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        max_survival_ticks: int,
        pipe_seed: int,
    ) -> GenerationRecord:
        improvement_rate = (
            (best_fitness - self._prev_best) / max(self._prev_best, 1.0)
        )
        rec = GenerationRecord(
            generation=generation,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            max_survival_ticks=max_survival_ticks,
            improvement_rate=improvement_rate,
            pipe_seed=pipe_seed,
        )
        self.history.append(rec)
        self._prev_best = best_fitness
        return rec

    def reset(self) -> None:
        self.history = []
        self._prev_best = 0.0
