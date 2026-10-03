"""Normalize relative loss importances using typical changes on this dataset."""

import math
import random
import statistics
from collections.abc import Iterable

from .datamodel import MetricName, ProblemData
from .utils import FALLBACK_DELTA_MAGNITUDE, MIN_DELTA_MAGNITUDE, compute_solution_metrics
from .variation_operator import initialize_random_tours, sample_neighbor

DEFAULT_CALIBRATION_SAMPLES = 256
DEFAULT_CALIBRATION_SEED = 0


def estimate_typical_metric_deltas(
    problem: ProblemData,
    metrics: Iterable[MetricName],
    *,
    samples: int = DEFAULT_CALIBRATION_SAMPLES,
    seed: int = DEFAULT_CALIBRATION_SEED,
) -> dict[MetricName, float]:
    """Return median significant absolute changes from an unconditional random walk.

    Accept every sampled neighbor, including worse ones. Reuse unchanged tour
    metrics and use a separate RNG so calibration does not advance the solver's
    RNG. Changes at or below MIN_DELTA_MAGNITUDE are ignored. A metric with no
    significant changes receives FALLBACK_DELTA_MAGNITUDE.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")

    selected_metrics = tuple(dict.fromkeys(metrics))
    deltas = {metric: [] for metric in selected_metrics}
    rng = random.Random(seed)
    tours = initialize_random_tours(problem, rng)
    metrics_by_tour, solution_metrics = compute_solution_metrics(problem, tours)

    for _ in range(samples):
        neighbor = sample_neighbor(problem, rng, tours)
        if neighbor is None:
            continue

        neighbor_tours, changed_agent_ids = neighbor
        neighbor_metrics_by_tour, neighbor_metrics = compute_solution_metrics(
            problem,
            neighbor_tours,
            previous_metrics=metrics_by_tour,
            changed_agent_ids=changed_agent_ids,
        )
        for metric in selected_metrics:
            delta = abs(
                getattr(neighbor_metrics, metric.value)
                - getattr(solution_metrics, metric.value)
            )
            if not math.isfinite(delta):
                raise ValueError(f"Non-finite delta for metric {metric.value!r}")
            if delta > MIN_DELTA_MAGNITUDE:
                deltas[metric].append(delta)

        tours = neighbor_tours
        metrics_by_tour = neighbor_metrics_by_tour
        solution_metrics = neighbor_metrics

    return {
        metric: statistics.median(values) if values else FALLBACK_DELTA_MAGNITUDE
        for metric, values in deltas.items()
    }


def calibrate_loss_weights(
    problem: ProblemData,
    importances: dict[MetricName, float],
    *,
    scales: dict[MetricName, float] | None = None,
) -> dict[MetricName, float]:
    """Convert importances to fixed weights: importance * distance_scale / scale.

    Distance stays at its configured importance. All other metrics are scaled
    relative to typical distance changes. Neither input dictionary is mutated.
    Supplied scales must include distance and every configured metric, with
    finite, strictly positive values.
    """
    if any(not math.isfinite(importance) for importance in importances.values()):
        raise ValueError("importances must be finite")

    metric_names = set(importances)
    metric_names.add(MetricName.TOTAL_DISTANCE)
    if scales is None:
        scales = estimate_typical_metric_deltas(problem, metric_names)
    for metric in metric_names:
        if metric not in scales or not math.isfinite(scales[metric]) or scales[metric] <= 0:
            raise ValueError(f"Scale for {metric.value!r} must be present, finite, and positive")
    distance_scale = scales[MetricName.TOTAL_DISTANCE]
    return {
        metric: importance * distance_scale / scales[metric]
        for metric, importance in importances.items()
    }
