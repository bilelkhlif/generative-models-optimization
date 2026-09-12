"""Combinatorial matching demo: assigning UAVs to disaster rescue sites so
that total travel distance is minimized, via the Hungarian algorithm.

This is a toy stand-in for the "optimization, algorithm design, or
combinatorial matching problems" skill explicitly asked for in the Mitacs GRI
"UAV-EV Collaborative Rescue Service Provision" project (Prof. Nan Chen,
University of Ottawa) -- that project's real setting also involves EVs and
reinforcement learning for the dynamic case; this demonstrates the static
assignment sub-problem that a full system would need to solve repeatedly.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def random_scenario(n_uavs: int = 6, n_sites: int = 6, seed: int = 0, area: float = 100.0):
    rng = np.random.default_rng(seed)
    uav_pos = rng.uniform(0, area, size=(n_uavs, 2))
    site_pos = rng.uniform(0, area, size=(n_sites, 2))
    return uav_pos, site_pos


def cost_matrix(uav_pos: np.ndarray, site_pos: np.ndarray) -> np.ndarray:
    diff = uav_pos[:, None, :] - site_pos[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def solve_assignment(uav_pos: np.ndarray, site_pos: np.ndarray):
    costs = cost_matrix(uav_pos, site_pos)
    row_ind, col_ind = linear_sum_assignment(costs)
    total_cost = costs[row_ind, col_ind].sum()
    return row_ind, col_ind, total_cost, costs


def greedy_baseline(uav_pos: np.ndarray, site_pos: np.ndarray):
    """Nearest-available-site greedy assignment, for comparison against the
    optimal Hungarian solution -- shows the actual gain from solving the
    problem properly rather than with a naive heuristic."""
    costs = cost_matrix(uav_pos, site_pos).copy()
    n = costs.shape[0]
    assigned_sites = set()
    assignment = [-1] * n
    order = np.argsort(costs.min(axis=1))  # handle "hardest" UAVs first
    for i in order:
        remaining = [j for j in range(n) if j not in assigned_sites]
        j = remaining[np.argmin(costs[i, remaining])]
        assignment[i] = j
        assigned_sites.add(j)
    total_cost = sum(costs[i, assignment[i]] for i in range(n))
    return assignment, total_cost


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    uav_pos, site_pos = random_scenario(n_uavs=10, n_sites=10, seed=2)
    row_ind, col_ind, opt_cost, costs = solve_assignment(uav_pos, site_pos)
    greedy_assignment, greedy_cost = greedy_baseline(uav_pos, site_pos)

    print(f"Hungarian (optimal) total distance: {opt_cost:.2f}")
    print(f"Greedy baseline total distance:     {greedy_cost:.2f}")
    print(f"Improvement: {100 * (greedy_cost - opt_cost) / greedy_cost:.1f}% shorter with optimal assignment")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, assignment, title, cost in zip(
        ax,
        [list(zip(row_ind, col_ind)), list(enumerate(greedy_assignment))],
        ["Hungarian (optimal)", "Greedy baseline"],
        [opt_cost, greedy_cost],
    ):
        a.scatter(*uav_pos.T, c="tab:blue", label="UAV", s=60, zorder=3)
        a.scatter(*site_pos.T, c="tab:red", marker="^", label="rescue site", s=80, zorder=3)
        for i, j in assignment:
            a.plot([uav_pos[i, 0], site_pos[j, 0]], [uav_pos[i, 1], site_pos[j, 1]], "k--", alpha=0.5)
        a.set_title(f"{title}\ntotal distance = {cost:.1f}")
        a.legend(loc="upper right", fontsize=8)
        a.set_xlim(-5, 105)
        a.set_ylim(-5, 105)
    plt.tight_layout()
    plt.savefig("assignment_comparison.png", dpi=150)
    print("Saved assignment_comparison.png")
