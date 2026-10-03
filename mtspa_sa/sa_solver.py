"""Simulated annealing for fixed-time multi-agent routing."""

import math
import random
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from .datamodel import (
    AgentId,
    LossConfig,
    LossSample,
    MetricName,
    ProblemData,
    Solution,
    Tours,
)
from .loss_calibration import calibrate_loss_weights
from .temperature_calibration import calibrate_temperature
from .utils import (
    DEFAULT_LOSS_CONFIG_ID,
    LOSS_CONFIGS,
    compute_loss,
    compute_solution_metrics,
)
from .variation_operator import initialize_random_tours, sample_neighbor


class SimulatedAnnealingSolver:
    def __init__(
        self,
        problem: ProblemData,
        *,
        seed: int | None = None,
        loss_config: LossConfig = LOSS_CONFIGS[DEFAULT_LOSS_CONFIG_ID],
        weights: dict[MetricName, float] | None = None,
    ) -> None:
        self.problem = problem
        self.rng = random.Random(seed)
        self.loss_config = loss_config
        self.weights = (
            calibrate_loss_weights(problem, loss_config.importances)
            if weights is None
            else weights.copy()
        )

    def build_solution(
        self,
        tours: Tours,
        *,
        previous: Solution | None = None,
        changed_agent_ids: tuple[AgentId, AgentId] | None = None,
    ) -> Solution:
        tour_metrics, solution_metrics = compute_solution_metrics(
            self.problem,
            tours,
            previous_metrics=previous.tour_metrics if previous is not None else None,
            changed_agent_ids=changed_agent_ids,
        )
        return Solution(
            tours=tours,
            tour_metrics=tour_metrics,
            solution_metrics=solution_metrics,
            loss=compute_loss(solution_metrics, self.weights),
        )

    def _sample_candidate(self, solution: Solution) -> Solution | None:
        neighbor = sample_neighbor(self.problem, self.rng, solution.tours)
        if neighbor is None:
            return None
        neighbor_tours, changed_agent_ids = neighbor
        return self.build_solution(
            neighbor_tours,
            previous=solution,
            changed_agent_ids=changed_agent_ids,
        )

    def optimize(self, steps: int) -> Solution:
        """Run one geometrically cooled Metropolis search."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        current = self.build_solution(initialize_random_tours(self.problem, self.rng))
        best = current
        temperature_config = calibrate_temperature(current, steps, self._sample_candidate)
        temperature = temperature_config.initial_temperature
        history = [LossSample(0, current.loss, best.loss)]
        sample_iterations = {(steps * percent + 99) // 100 for percent in range(1, 101)}

        for iteration in tqdm(range(1, steps + 1), desc="SA steps", unit="step"):
            candidate = self._sample_candidate(current)
            if candidate is not None:
                loss_change = candidate.loss - current.loss
                accept = loss_change <= 0 or self.rng.random() < math.exp(
                    -loss_change / temperature
                )
                if accept:
                    current = candidate
                    if current.loss < best.loss:
                        best = current
            temperature *= temperature_config.cooling_rate
            if iteration in sample_iterations:
                history.append(LossSample(iteration, current.loss, best.loss))
        return replace(best, loss_history=history)

    def optimize_parallel(self, steps: int, n_runs: int) -> Solution:
        """Run independently seeded searches in worker processes."""
        if steps <= 0 or n_runs <= 0:
            raise ValueError("steps and n_runs must be positive")
        solvers = [
            SimulatedAnnealingSolver(
                self.problem,
                seed=self.rng.getrandbits(64),
                loss_config=self.loss_config,
                weights=self.weights,
            )
            for _ in range(n_runs)
        ]
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(solver.optimize, steps)
                for solver in solvers
            ]
            solutions = [
                future.result()
                for future in tqdm(
                    as_completed(futures),
                    total=n_runs,
                    desc="SA runs",
                    unit="run",
                )
            ]
        return min(solutions, key=lambda solution: solution.loss)
