"""Routing data models, metrics, and solver configuration."""

from dataclasses import dataclass, field
from enum import StrEnum


type AgentId = int
type NodeId = int
type Tour = list[NodeId]
type Tours = list[Tour]
type Matrix = list[list[int]]


@dataclass(frozen=True, slots=True)
class Node:
    id: NodeId
    time: int
    duration: int
    kind: str
    agent_id: AgentId | None
    gain: int
    latitude: float | None = None
    longitude: float | None = None


@dataclass(frozen=True, slots=True)
class Agent:
    id: AgentId
    name: str
    start_time: int
    end_time: int


@dataclass(slots=True)
class ProblemData:
    nodes: list[Node]
    agents: list[Agent]
    distances: Matrix
    travel_times: Matrix
    node_times: list[int] = field(init=False, repr=False)
    node_durations: list[int] = field(init=False, repr=False)
    node_gains: list[int] = field(init=False, repr=False)
    node_is_home: list[bool] = field(init=False, repr=False)
    appointment_ids: list[NodeId] = field(init=False, repr=False)
    agent_start_times: list[int] = field(init=False, repr=False)
    agent_end_times: list[int] = field(init=False, repr=False)
    agent_home_nodes: list[NodeId] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        node_count = len(self.nodes)
        agent_count = len(self.agents)
        if agent_count < 2:
            raise ValueError("Simulated annealing requires at least two agents")
        if [node.id for node in self.nodes] != list(range(node_count)):
            raise ValueError("Node IDs must be consecutive and start at 0")
        if [agent.id for agent in self.agents] != list(range(agent_count)):
            raise ValueError("Agent IDs must be consecutive and start at 0")
        for name, matrix in (("Distance", self.distances), ("Travel-time", self.travel_times)):
            if len(matrix) != node_count or any(len(row) != node_count for row in matrix):
                raise ValueError(f"{name} matrix shape does not match the nodes")

        home_nodes = [node for node in self.nodes if node.kind == "home"]
        if any(node.agent_id is None or not 0 <= node.agent_id < agent_count for node in home_nodes):
            raise ValueError("Home nodes must reference valid agent IDs")
        homes = {node.agent_id: node.id for node in home_nodes}
        if len(home_nodes) != agent_count or len(homes) != agent_count:
            raise ValueError("Each agent must have exactly one home node")
        self.agent_home_nodes = [homes[agent.id] for agent in self.agents]

        self.node_times = [node.time for node in self.nodes]
        self.node_durations = [node.duration for node in self.nodes]
        self.node_gains = [node.gain for node in self.nodes]
        self.node_is_home = [node.kind == "home" for node in self.nodes]
        self.appointment_ids = [
            node.id for node in self.nodes if node.kind != "home"
        ]
        self.agent_start_times = [agent.start_time for agent in self.agents]
        self.agent_end_times = [agent.end_time for agent in self.agents]


@dataclass(frozen=True, slots=True)
class TourMetrics:
    distance: int
    gain: int
    elapsed_time: int
    gain_per_hour: float
    lateness: int
    waiting_time: int
    overtime: int


@dataclass(frozen=True, slots=True)
class SolutionMetrics:
    total_distance: int
    distance_std: float
    gain_per_hour_std: float
    total_lateness: int
    total_waiting_time: int
    total_overtime: int


@dataclass(frozen=True, slots=True)
class LossSample:
    iteration: int
    loss: float
    best_loss: float


@dataclass(frozen=True, slots=True)
class Solution:
    tours: Tours
    tour_metrics: list[TourMetrics]
    solution_metrics: SolutionMetrics
    loss: float
    loss_history: list[LossSample] = field(default_factory=list)


class MetricName(StrEnum):
    TOTAL_DISTANCE = "total_distance"
    DISTANCE_STD = "distance_std"
    GAIN_PER_HOUR_STD = "gain_per_hour_std"
    TOTAL_LATENESS = "total_lateness"
    TOTAL_WAITING_TIME = "total_waiting_time"
    TOTAL_OVERTIME = "total_overtime"


@dataclass(frozen=True, slots=True)
class LossConfig:
    name: str
    importances: dict[MetricName, float]


@dataclass(frozen=True, slots=True)
class TemperatureConfig:
    initial_temperature: float
    final_temperature: float
    cooling_rate: float
