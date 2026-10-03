"""Initialize tours and generate neighbors by transferring or swapping visits."""

import random
from bisect import insort_right

from datamodel import AgentId, ProblemData, Tour, Tours


SWAP_PROBABILITY = 0.5


def initialize_random_tours(problem: ProblemData, rng: random.Random) -> Tours:
    """Assign each appointment randomly, then sort routes by appointment time."""
    appointments_by_agent = [[] for _ in problem.agents]
    for appointment_id in problem.appointment_ids:
        agent_id = rng.randrange(len(problem.agents))
        appointments_by_agent[agent_id].append(appointment_id)

    tours = []
    for agent_id, appointments in enumerate(appointments_by_agent):
        appointments.sort(
            key=lambda node_id: (problem.node_times[node_id], node_id)
        )
        home = problem.agent_home_nodes[agent_id]
        tours.append([home, *appointments, home])
    return tours


def _transfer_appointment(
    problem: ProblemData,
    rng: random.Random,
    donor_tour: Tour,
    receiver_tour: Tour,
) -> None:
    """Move one random appointment, preserving chronological order."""
    appointment_index = rng.randrange(1, len(donor_tour) - 1)
    appointment = donor_tour.pop(appointment_index)
    insort_right(
        receiver_tour,
        appointment,
        lo=1,
        hi=len(receiver_tour) - 1,
        key=lambda node_id: (problem.node_times[node_id], node_id),
    )


def _swap_appointments(
    problem: ProblemData,
    rng: random.Random,
    first_tour: Tour,
    second_tour: Tour,
) -> None:
    """Exchange one appointment between two routes, preferring equal times."""
    first_index = rng.randrange(1, len(first_tour) - 1)
    first_appointment = first_tour[first_index]
    first_time = problem.node_times[first_appointment]
    same_time_indices = [
        index
        for index in range(1, len(second_tour) - 1)
        if problem.node_times[second_tour[index]] == first_time
    ]
    second_index = (
        rng.choice(same_time_indices)
        if same_time_indices
        else rng.randrange(1, len(second_tour) - 1)
    )
    first_tour.pop(first_index)
    second_appointment = second_tour.pop(second_index)
    for tour, appointment in (
        (first_tour, second_appointment),
        (second_tour, first_appointment),
    ):
        insort_right(
            tour,
            appointment,
            lo=1,
            hi=len(tour) - 1,
            key=lambda node_id: (problem.node_times[node_id], node_id),
        )


def sample_neighbor(
    problem: ProblemData,
    rng: random.Random,
    tours: Tours,
) -> tuple[Tours, tuple[AgentId, AgentId]] | None:
    """Copy only two selected routes and apply a swap or transfer."""
    donor_id, receiver_id = rng.sample(range(len(tours)), k=2)
    if len(tours[donor_id]) <= 2:
        donor_id, receiver_id = receiver_id, donor_id
    if len(tours[donor_id]) <= 2:
        return None

    donor_tour = tours[donor_id].copy()
    receiver_tour = tours[receiver_id].copy()
    if len(receiver_tour) > 2 and rng.random() < SWAP_PROBABILITY:
        _swap_appointments(problem, rng, donor_tour, receiver_tour)
    else:
        _transfer_appointment(problem, rng, donor_tour, receiver_tour)

    neighbor = tours.copy()
    neighbor[donor_id] = donor_tour
    neighbor[receiver_id] = receiver_tour
    return neighbor, (donor_id, receiver_id)
