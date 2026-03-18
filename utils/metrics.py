from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationMetrics:
    policy: str
    average_score: float
    score_per_move: float
    variance: float
    episodes: int


def summarize_scores(policy: str, scores: list[float], max_moves: int) -> EvaluationMetrics:
    values = np.array(scores, dtype=np.float64)
    avg = float(values.mean()) if len(values) else 0.0
    return EvaluationMetrics(
        policy=policy,
        average_score=avg,
        score_per_move=avg / float(max_moves),
        variance=float(values.var()) if len(values) else 0.0,
        episodes=len(scores),
    )
