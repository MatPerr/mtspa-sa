# MTSPA-SA

This repository is a small, standalone example of simulated annealing applied to
appointment routing. It keeps the optimization code easy to explore: load a
sample dataset, run a search, and inspect or plot the resulting tours, without
starting a web app or querying a routing service.

Assign appointments to multiple agents, each with a home and working hours.
Appointments have fixed locations, scheduled start times, and durations;
travel between them takes time and adds distance.

The search state is `tours`: a list of routes indexed by agent ID. Each route
contains node IDs, starts and ends at that agent's home, and orders appointments
by scheduled time. For example, with homes `0` and `1`:

```python
tours = [
    [0, 2, 4, 0],  # Agent 0: home → appointment 2 → appointment 4 → home
    [1, 3, 5, 1],  # Agent 1: home → appointment 3 → appointment 5 → home
]
```

Every appointment appears once; an empty route is `[home, home]`. A neighbor
transfers or swaps appointments between two routes. `Solution` bundles the tours
with `tour_metrics`, `solution_metrics`, and `loss`; the solver tracks the current
solution and the best one found.

The goal is to find tours with the lowest weighted loss: minimize distance,
balance agents' distances or hourly pay, or reduce waiting time, depending on
the chosen objective. Simulated annealing explores transfers and swaps while
retaining the best solution found. Timing violations are reported; the current
profiles penalize lateness and give overtime zero weight, so inspect both
before treating a solution as feasible.

Install dependencies and run a reproducible search:

```bash
uv sync --locked
uv run mtspa-sa data/belgium_real_estate.json --steps 50000 --seed 0
```

Use `--runs 4` for parallel searches and
`--loss-config fair_distance` to choose an objective. Available objectives are
`shortest_distance`, `fair_hourly_pay`, `fair_distance`, and `maximum_uptime`.
Relative metric importances are editable in [`loss_config.yaml`](loss_config.yaml).
[`loss_calibration.py`](mtspa_sa/loss_calibration.py) converts them to dataset-specific weights
once when constructing the solver. It samples 256 random moves with a separate,
fixed-seed RNG and estimates each metric's typical change as the median nonzero
absolute change (falling back to 1 if none is observed):

```text
weight = importance × distance_scale / metric_scale
```

Distance keeps its configured importance of 1. Weights stay fixed throughout
optimization and are shared by parallel runs. This normalizes the objectives
across datasets; it does not guarantee feasibility or optimality.

Display the resulting tours and optimization losses with [`reporting.py`](mtspa_sa/reporting.py):

```bash
uv run mtspa-sa data/corsica_nurses.json --steps 50000 --plot
```

Each agent has a color; square markers are homes. Lines connect visits directly
in route order rather than following roads. The right subplot shows current and best
loss, sampled every 1% of iterations (including the start and finish). For parallel
runs, it shows the history of the winning run.
In Python, `plot_tours(problem, solution.tours, solution.loss_history)`
returns a [Matplotlib figure](https://matplotlib.org/stable/api/figure_api.html)
that you can display or save.

Bundled datasets contain distance matrices in metres and travel-time matrices
in seconds, so no routing service is needed. JSON and gzip-compressed JSON are supported.

| Dataset | Agents | Appointments |
| --- | ---: | ---: |
| `data/belgium_real_estate.json` — Belgium real estate | 7 | 21 |
| `data/corsica_nurses.json` — Ajaccio patient visits | 2 | 15 |
| `data/paris_lunch_deliveries.json` — Paris lunch deliveries | 8 | 80 |
| `data/paris_dinner_deliveries.json.gz` — Paris dinner deliveries | 30 | 300 |

Pass any dataset path in place of `data/belgium_real_estate.json`.

All Python code lives in the `mtspa_sa/` package; sample data and the editable
loss config stay at the root:

```text
mtspa-sa/
├── mtspa_sa/
│   ├── __init__.py
│   ├── __main__.py              # CLI
│   ├── sa_solver.py             # Algorithm
│   ├── datamodel.py
│   ├── loss_calibration.py
│   ├── temperature_calibration.py
│   ├── variation_operator.py
│   ├── reporting.py
│   └── utils.py
├── data/
├── loss_config.yaml
├── README.md
├── pyproject.toml
└── uv.lock
```

`uv run python -m mtspa_sa` is equivalent to `uv run mtspa-sa`.
Run these commands from the repository root, or supply a dataset path when
running elsewhere. Wheels include a copy of the default loss config; editable
installs read the root YAML.

The CLI lives in [`__main__.py`](mtspa_sa/__main__.py). The search runs in
[`sa_solver.py`](mtspa_sa/sa_solver.py), with initialization and moves in
[`variation_operator.py`](mtspa_sa/variation_operator.py), metric calculations and
loading in [`utils.py`](mtspa_sa/utils.py), models in [`datamodel.py`](mtspa_sa/datamodel.py),
loss calibration in [`loss_calibration.py`](mtspa_sa/loss_calibration.py), and
temperature calibration in [`temperature_calibration.py`](mtspa_sa/temperature_calibration.py).

The call flow for one search:

```text
main()
├── load_problem()
├── SimulatedAnnealingSolver(...)
│   └── calibrate_loss_weights()
│       └── estimate_typical_metric_deltas()
│           ├── initialize_random_tours()
│           └── Sample neighbors and compute metrics (256 attempts)
├── optimize()
│   ├── initialize_random_tours()
│   ├── build_solution()                         # Initial solution
│   │   ├── compute_solution_metrics()
│   │   │   ├── _compute_tour_metrics()          # All routes initially; two for a neighbor
│   │   │   └── _aggregate_metrics()
│   │   │       └── _std_from_squares()
│   │   └── compute_loss()
│   ├── calibrate_temperature()
│   │   └── _estimate_typical_loss_delta()
│   │       └── for each calibration sample
│   │           └── _sample_candidate()          # Same call chain as below
│   ├── for each annealing step
│   │   ├── _sample_candidate()
│   │   │   ├── sample_neighbor()
│   │   │   │   ├── _transfer_appointment()      # Either transfer
│   │   │   │   └── _swap_appointments()         # Or swap
│   │   │   └── build_solution()                 # If a neighbor was generated
│   │   ├── if a candidate was generated
│   │   │   ├── Accept or reject candidate
│   │   │   └── Update best solution if improved
│   │   └── Cool temperature
│   └── Return best solution
└── plot_tours()                                 # If --plot was supplied
```

`load_loss_configs()` reads the YAML profiles when `utils.py` is imported.
With `--runs`, `main()` calls `optimize_parallel()`, which runs `optimize()` in
worker processes and returns the solution with the lowest loss.
