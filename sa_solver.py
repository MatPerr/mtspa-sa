"""Simulated annealing for fixed-time multi-agent routing and its CLI."""

import argparse
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from datamodel import (
    AgentId,
    LossConfig,
    MetricName,
    ProblemData,
    Solution,
    Tours,
)
from loss_calibration import calibrate_loss_weights
from temperature_calibration import calibrate_temperature
from utils import (
    DEFAULT_LOSS_CONFIG_ID,
    LOSS_CONFIGS,
    compute_loss,
    compute_solution_metrics,
    load_problem,
)
from variation_operator import initialize_random_tours, sample_neighbor


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

        for _ in tqdm(range(steps), desc="SA steps", unit="step"):
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
        return best

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal simulated annealing"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("data/belgium_real_estate.json"),
    )
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="Display a plot of the resulting tours")
    parser.add_argument(
        "--loss-config",
        choices=tuple(LOSS_CONFIGS),
        default=DEFAULT_LOSS_CONFIG_ID,
    )
    arguments = parser.parse_args()
    if arguments.steps <= 0 or arguments.runs <= 0:
        parser.error("--steps and --runs must be positive")

    problem = load_problem(arguments.dataset)
    solver = SimulatedAnnealingSolver(
        problem,
        seed=arguments.seed,
        loss_config=LOSS_CONFIGS[arguments.loss_config],
    )
    started = time.perf_counter()
    if arguments.runs == 1:
        solution = solver.optimize(arguments.steps)
    else:
        solution = solver.optimize_parallel(arguments.steps, arguments.runs)
    elapsed_seconds = time.perf_counter() - started

    print(f"Objective: {solver.loss_config.name}")
    print(
        f"Elapsed: {elapsed_seconds:.3f} s "
        f"({arguments.runs} run(s), {arguments.steps:,} steps each)"
    )
    print(f"Loss: {solution.loss:.3f}")
    print(
        "Total distance: "
        f"{solution.solution_metrics.total_distance / 1000:.3f} km"
    )
    print(f"Distance std: {solution.solution_metrics.distance_std / 1000:.3f} km")
    print(f"Gain per hour std: {solution.solution_metrics.gain_per_hour_std:.3f}")
    print(f"Total lateness: {solution.solution_metrics.total_lateness} s")
    print(f"Total waiting: {solution.solution_metrics.total_waiting_time} s")
    print(f"Total overtime: {solution.solution_metrics.total_overtime} s")
    print("Tours (node IDs, including homes):")
    for agent, tour in zip(problem.agents, solution.tours, strict=True):
        print(f"  {agent.name}: {' -> '.join(map(str, tour))}")

    if arguments.plot:
        import matplotlib.pyplot as plt

        from reporting import plot_tours

        plot_tours(problem, solution.tours)
        plt.show()


if __name__ == "__main__":
    main()
