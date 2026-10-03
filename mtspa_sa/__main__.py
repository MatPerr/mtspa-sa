"""Command-line interface for the standalone simulated annealing solver."""

import argparse
import time
from pathlib import Path

from .sa_solver import SimulatedAnnealingSolver
from .utils import DEFAULT_LOSS_CONFIG_ID, LOSS_CONFIGS, load_problem


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
    parser.add_argument("--plot", action="store_true", help="Display tours and optimization losses")
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

        from .reporting import plot_tours

        plot_tours(problem, solution.tours, solution.loss_history)
        plt.show()


if __name__ == "__main__":
    main()
