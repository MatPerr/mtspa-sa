"""Dataset loading, objective configuration, and route evaluation."""

import gzip
import json
import math
from itertools import pairwise
from pathlib import Path

import yaml

from datamodel import (
    Agent,
    AgentId,
    LossConfig,
    MetricName,
    Node,
    ProblemData,
    SolutionMetrics,
    Tour,
    TourMetrics,
    Tours,
)

MIN_DELTA_MAGNITUDE = 1e-12
FALLBACK_DELTA_MAGNITUDE = 1.0


def load_loss_configs(filepath: str | Path) -> tuple[dict[str, LossConfig], str]:
    """Load relative importances and require every metric in every profile."""
    with Path(filepath).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    expected_metrics = {metric.value for metric in MetricName}
    configs = {}
    for config_id, profile in data["profiles"].items():
        raw_importances = profile["importances"]
        if set(raw_importances) != expected_metrics:
            raise ValueError(f"Loss profile {config_id!r} must define exactly {sorted(expected_metrics)}")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in raw_importances.values()):
            raise ValueError(f"Loss profile {config_id!r} importances must be finite numbers")
        configs[config_id] = LossConfig(
            name=profile["name"],
            importances={metric: float(raw_importances[metric.value]) for metric in MetricName},
        )

    default_id = data["default"]
    if default_id not in configs:
        raise ValueError(f"Unknown default loss profile {default_id!r}")
    return configs, default_id


LOSS_CONFIG_PATH = Path(__file__).with_name("loss_config.yaml")
LOSS_CONFIGS, DEFAULT_LOSS_CONFIG_ID = load_loss_configs(LOSS_CONFIG_PATH)


def load_problem(filepath: str | Path) -> ProblemData:
    """Load a sample containing agents, nodes, and complete D/T matrices."""
    path = Path(filepath)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        data = json.load(file)
    nodes = [
        Node(
            id=int(item["id"]),
            time=int(item["time"]),
            duration=int(item["duration"]),
            kind=str(item["type"]),
            agent_id=(
                None
                if item["agent_id"] is None
                else int(item["agent_id"])
            ),
            gain=int(item.get("gain", 0)),
            latitude=float(item["lat"]) if "lat" in item else None,
            longitude=float(item["lng"]) if "lng" in item else None,
        )
        for item in sorted(data["nodes"], key=lambda item: int(item["id"]))
    ]
    agents = [
        Agent(
            id=int(item["id"]),
            name=str(item["name"]),
            start_time=int(item["startTime"]),
            end_time=int(item["endTime"]),
        )
        for item in sorted(data["agents"], key=lambda item: int(item["id"]))
    ]
    distances = [list(map(int, row)) for row in data["D"]]
    travel_times = [list(map(int, row)) for row in data["T"]]
    return ProblemData(nodes, agents, distances, travel_times)


def compute_solution_metrics(
    problem: ProblemData,
    tours: Tours,
    *,
    previous_metrics: list[TourMetrics] | None = None,
    changed_agent_ids: tuple[AgentId, AgentId] | None = None,
) -> tuple[list[TourMetrics], SolutionMetrics]:
    """Return per-tour metrics and their solution-wide aggregate.

    Evaluate every tour initially; reuse previous metrics for unchanged tours
    when evaluating a neighbor.
    """
    if previous_metrics is None:
        if changed_agent_ids is not None:
            raise ValueError("changed_agent_ids requires previous_metrics")
        tour_metrics = [
            _compute_tour_metrics(problem, agent_id, tour)
            for agent_id, tour in enumerate(tours)
        ]
    else:
        if changed_agent_ids is None:
            raise ValueError(
                "changed_agent_ids is required with previous_metrics"
            )
        tour_metrics = previous_metrics.copy()
        for agent_id in changed_agent_ids:
            tour_metrics[agent_id] = _compute_tour_metrics(
                problem,
                agent_id,
                tours[agent_id],
            )
    return tour_metrics, _aggregate_metrics(tour_metrics)


def _compute_tour_metrics(problem: ProblemData, agent_id: AgentId, tour: Tour) -> TourMetrics:
    """Simulate one route, allowing lateness and overtime as soft penalties."""
    distance = 0
    lateness = 0
    waiting_time = 0
    overtime = 0
    current_time = problem.agent_start_times[agent_id]

    for origin_id, destination_id in pairwise(tour):
        distance += problem.distances[origin_id][destination_id]
        arrival_time = (
            current_time + problem.travel_times[origin_id][destination_id]
        )
        if problem.node_is_home[destination_id]:
            overtime = max(
                0,
                arrival_time - problem.agent_end_times[agent_id],
            )
            current_time = arrival_time
            continue

        scheduled_time = problem.node_times[destination_id]
        time_difference = arrival_time - scheduled_time
        waiting_time += max(0, -time_difference)
        lateness += max(0, time_difference)
        current_time = (
            max(arrival_time, scheduled_time)
            + problem.node_durations[destination_id]
        )

    gain = sum(problem.node_gains[node_id] for node_id in tour[1:-1])
    elapsed_time = current_time - problem.agent_start_times[agent_id]
    return TourMetrics(
        distance=distance,
        gain=gain,
        elapsed_time=elapsed_time,
        gain_per_hour=(
            gain / (elapsed_time / 3600) if elapsed_time else 0.0
        ),
        lateness=lateness,
        waiting_time=waiting_time,
        overtime=overtime,
    )


def _std_from_squares(total: int | float, sum_of_squares: int | float, count: int) -> float:
    """Calculate standard deviation across all agents from sums and count."""
    mean = total / count
    variance = sum_of_squares / count - mean * mean
    return math.sqrt(max(0.0, variance))


def _aggregate_metrics(tour_metrics: list[TourMetrics]) -> SolutionMetrics:
    """Aggregate all cross-tour metrics in one pass."""
    count = len(tour_metrics)
    total_distance = 0
    distance_squares = 0
    gain_per_hour_total = 0.0
    gain_per_hour_squares = 0.0
    total_lateness = 0
    total_waiting_time = 0
    total_overtime = 0
    for metrics in tour_metrics:
        total_distance += metrics.distance
        distance_squares += metrics.distance * metrics.distance
        gain_per_hour_total += metrics.gain_per_hour
        gain_per_hour_squares += metrics.gain_per_hour**2
        total_lateness += metrics.lateness
        total_waiting_time += metrics.waiting_time
        total_overtime += metrics.overtime
    return SolutionMetrics(
        total_distance=total_distance,
        distance_std=_std_from_squares(
            total_distance,
            distance_squares,
            count,
        ),
        gain_per_hour_std=_std_from_squares(
            gain_per_hour_total,
            gain_per_hour_squares,
            count,
        ),
        total_lateness=total_lateness,
        total_waiting_time=total_waiting_time,
        total_overtime=total_overtime,
    )


def compute_loss(solution_metrics: SolutionMetrics, weights: dict[MetricName, float]) -> float:
    return sum(
        weight * getattr(solution_metrics, metric.value)
        for metric, weight in weights.items()
    )
