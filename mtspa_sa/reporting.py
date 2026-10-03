"""Plot tours as straight connections between geographic coordinates."""

import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .datamodel import LossSample, ProblemData, Tours


def plot_tours(
    problem: ProblemData,
    tours: Tours,
    loss_history: list[LossSample] | None = None,
) -> Figure:
    """Plot tours and, when supplied, sampled current and best losses."""
    if any(node.latitude is None or node.longitude is None for node in problem.nodes):
        raise ValueError("Plotting requires lat/lng coordinates for every node")

    if loss_history is None:
        figure, axes = plt.subplots(figsize=(10, 7), layout="constrained")
    else:
        figure, (axes, loss_axes) = plt.subplots(
            1, 2, figsize=(17, 7), layout="constrained"
        )
        iterations = [sample.iteration for sample in loss_history]
        loss_axes.plot(
            iterations, [sample.loss for sample in loss_history],
            "o-", markersize=3, label="Current loss",
        )
        loss_axes.plot(
            iterations, [sample.best_loss for sample in loss_history],
            "o-", markersize=3, label="Best loss",
        )
        loss_axes.set(xlabel="Iteration", ylabel="Loss", title="Optimization loss")
        loss_axes.grid(alpha=0.2)
        loss_axes.legend()
    colors = plt.get_cmap("turbo", len(problem.agents))
    for agent, tour in zip(problem.agents, tours, strict=True):
        nodes = [problem.nodes[node_id] for node_id in tour]
        color = colors(agent.id)
        axes.plot(
            [node.longitude for node in nodes],
            [node.latitude for node in nodes],
            "o-",
            color=color,
            markersize=4,
            linewidth=1,
            label=agent.name,
        )
        home = problem.nodes[problem.agent_home_nodes[agent.id]]
        axes.scatter(home.longitude, home.latitude, marker="s", s=90, color=color, edgecolors="black", zorder=3)

    mean_latitude = sum(node.latitude for node in problem.nodes) / len(problem.nodes)
    axes.set_aspect(1 / math.cos(math.radians(mean_latitude)))
    axes.set(xlabel="Longitude", ylabel="Latitude", title="Agent tours (squares = homes)")
    axes.ticklabel_format(useOffset=False)
    axes.grid(alpha=0.2)
    axes.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    return figure
