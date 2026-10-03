"""Plot tours as straight connections between geographic coordinates."""

import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from datamodel import ProblemData, Tours


def plot_tours(problem: ProblemData, tours: Tours) -> Figure:
    """Return a figure with one color per agent and square markers for homes."""
    if any(node.latitude is None or node.longitude is None for node in problem.nodes):
        raise ValueError("Plotting requires lat/lng coordinates for every node")

    figure, axes = plt.subplots(figsize=(10, 7), layout="constrained")
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
