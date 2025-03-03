from typing import Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from modules.config import CLUSTER_VALUE_DLA, CONCENTRATION_VALUE
from modules.grid import initialize_grid


class Diffusion:
    def __init__(self, N: int, eta: float, initial_point: str = "bottom"):
        """
        Params:
        -------
        - N: int, size of the grid
        - eta: float, parameter for the probability calculation
        - initial_point: str, where to start the cluster
        """
        self.grid = initialize_grid(N)
        self.N = N
        self.eta = eta

        self.cluster: Set[Tuple[int, int]] = set()
        self.perimeter: Set[Tuple[int, int]] = set()

        if initial_point == "top":
            coords = (0, N // 2)
            self.add_to_cluster(coords)
            self.grid[-1, :] = CONCENTRATION_VALUE
        elif initial_point == "bottom":
            coords = (N - 1, N // 2)
            self.grid = np.array(
                [[1 - row / (N - 1) for col in range(N)] for row in range(N)]
            )  # initial guess for SOR
            self.add_to_cluster(coords)
        elif initial_point == "center":
            coords = (N // 2, N // 2)
            self.add_to_cluster(coords)
            self.grid[0, :] = CONCENTRATION_VALUE
            self.grid[-1, :] = CONCENTRATION_VALUE
            self.grid[:, 0] = CONCENTRATION_VALUE
            self.grid[:, -1] = CONCENTRATION_VALUE
        else:
            raise ValueError(
                "Invalid initial point, choose from 'bottom', 'top' or 'center'"
            )

        assert self.grid[coords] == CLUSTER_VALUE_DLA
        assert self.cluster == {coords}
        assert self.perimeter == self.get_neighbours(coords)

    def add_to_cluster(self, coords: Tuple[int, int]):
        """
        Add a point to the cluster and update the perimeter

        Params:
        -------
        - coords: Tuple[int, int], coordinates of the point to add
        """

        x, y = coords
        assert x >= 0 and x < self.N
        assert y >= 0 and y < self.N

        self.cluster.add(coords)
        self.grid[coords] = CLUSTER_VALUE_DLA
        self.perimeter -= self.cluster

        # Add neighbours to the perimeter
        neighbours = self.get_neighbours(coords)
        neighbours -= self.cluster
        self.perimeter |= neighbours

        assert coords not in self.perimeter
        assert coords in self.cluster

    def get_neighbours(self, coords: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        Get the neighbours of a point

        Params:
        -------
        - coords: Tuple[int, int], coordinates of the point

        Returns:
        --------
        - neighbours: Set[Tuple[int, int]], set of coordinates of the neighbours
        """
        x, y = coords
        assert x >= 0 and x < self.N
        assert y >= 0 and y < self.N

        neighbours = set()
        if x > 0:
            neighbours.add((x - 1, y))
        if x < self.N - 1:
            neighbours.add((x + 1, y))
        if y > 0:
            neighbours.add((x, y - 1))
        if y == 0:
            neighbours.add((x, self.N - 2))
        if y < self.N - 1:
            neighbours.add((x, y + 1))
        if y == self.N - 1:
            neighbours.add((x, 1))

        assert 2 <= len(neighbours) <= 4
        assert coords not in neighbours
        assert isinstance(neighbours, set)
        return neighbours

    def get_perimeter_size(self):
        return len(self.perimeter)

    def get_width(self):
        return max([coords[1] for coords in self.cluster]) - min(
            [coords[1] for coords in self.cluster]
        )

    def get_height(self):
        return max([coords[0] for coords in self.cluster]) - min(
            [coords[0] for coords in self.cluster]
        )

    def plot(self, save: bool = False, filename: str = "dla.png"):
        fig, ax = plt.subplots()
        im = ax.imshow(self.grid, cmap="Blues")
        plt.colorbar(im)

        x_points = [coords[1] for coords in self.cluster]
        y_points = [coords[0] for coords in self.cluster]
        ax.scatter(x_points, y_points, color="black", s=2)

        if save:
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        plt.show()

    def grow_cluster(self, omega: float = 1.8):
        """
        Grow the cluster by adding a random point from the perimeter
        """
        result_sor, sor_iters = successive_over_relaxation(
            self.grid, self.cluster, omega
        )
        self.grid = result_sor

        boundary_coords = [coords for coords in self.perimeter]
        boundary_concentration = [
            self.grid[coords] ** self.eta if self.grid[coords] >= 0 else 0
            for coords in boundary_coords
        ]
        total_boundary_concentration = sum(boundary_concentration)

        # Pick one cell randomly
        new_cell_idx = np.random.choice(
            [i for i in range(len(self.perimeter))],
            size=1,
            p=boundary_concentration / total_boundary_concentration,
        )[0]
        new_coords = boundary_coords[new_cell_idx]

        self.add_to_cluster(new_coords)

        # Periodic conditions
        if new_coords[1] == 0:
            self.add_to_cluster((new_coords[0], self.N - 1))
        elif new_coords[1] == self.N - 1:
            self.add_to_cluster((new_coords[0], 0))

        return sor_iters

    def run_simulation(self, steps: int = 1000, omega: float = 1.8):
        """Runs the DLA simulation for a given number of steps."""
        sor_iters = 0
        for i in range(steps):
            sor_iters += self.grow_cluster(omega)

        return sor_iters


def successive_over_relaxation(
    grid: np.ndarray,
    cluster: Set[Tuple[int, int]],
    omega: float | None = 1.8,
    epsilon: float | None = 1e-5,
    max_iterations: int | None = 5000,
) -> Tuple[list, list, int]:
    """
    Run the Successive Under Relaxation method to solve the time independent diffusion equation

    Params:
    -------
    - grid (np.ndarray): The initial spatial grid
    - cluster (Set[Tuple[int, int]]): The coordinates of the cluster
    - epsilon (float, optional): The convergence criterion. Defaults to 1e-5.
    - max_iterations (int, optional): The maximum number of iterations. Defaults to 100000.
    - omega (float, optional): The relaxation factor. Defaults to 1.8

    Returns:
    --------
    - results (List[np.ndarray]): A list of spatial grids at each iteration
    - k (int): The number of iterations
    """
    grid = grid.copy()
    N = grid.shape[0] - 1

    delta = float("inf")
    k = 0

    while delta > epsilon and k < max_iterations:
        delta = 0

        # First column
        for i in range(1, N):
            if (i, 0) in cluster:
                continue

            old_cell = grid[i][0].copy()
            grid[i][0] = (
                omega
                / 4
                * (grid[i + 1][0] + grid[i - 1][0] + grid[i][1] + grid[i][N - 1])
                + (1 - omega) * grid[i][0]
            )

            if np.abs(grid[i][0] - old_cell) > delta:
                delta = np.abs(grid[i][0] - old_cell)

        for j in range(1, N):
            for i in range(1, N):
                if (i, j) in cluster:
                    continue

                old_cell = grid[i][j].copy()
                grid[i][j] = (
                    omega
                    / 4
                    * (
                        grid[i + 1][j]
                        + grid[i - 1][j]
                        + grid[i][j + 1]
                        + grid[i][j - 1]
                    )
                    + (1 - omega) * grid[i][j]
                )

                delta = max(delta, np.abs(grid[i][j] - old_cell))

        # Last column
        for i in range(1, N):
            if (i, N) in cluster:
                continue

            old_cell = grid[i][N].copy()
            grid[i][N] = (
                omega
                / 4
                * (grid[i + 1][N] + grid[i - 1][N] + grid[i][0] + grid[i][N - 1])
                + (1 - omega) * grid[i][N]
            )

            if np.abs(grid[i][N] - old_cell) > delta:
                delta = np.abs(grid[i][N] - old_cell)

        k += 1

    return grid, k


def simulate_different_omegas(
    eta: float = 1, omegas: list[float] = [1.0, 1.4, 1.8, 1.9]
):
    grid_size = 100
    num_iterations = 50

    results = np.zeros(len(omegas))
    for j, omega in enumerate(omegas):
        diffusion = Diffusion(grid_size, eta, initial_point="bottom")

        results[j] = diffusion.run_simulation(num_iterations, omega)

    return results


def compare_omegas(
    eta: float = 1,
    omegas: list[float] = [1.0, 1.4, 1.8, 1.9],
    num_simulations: int = 20,
):
    results = Parallel(n_jobs=-2)(
        delayed(simulate_different_omegas)(eta, omegas) for _ in range(num_simulations)
    )

    return np.array(results)


def plot_omega_comparison(
    results, omegas, eta, save=False, filename="omega_comparison.png"
):
    plt.figure(figsize=(8, 5))
    plt.boxplot(results, tick_labels=omegas)
    plt.xlabel(r"$\omega$")
    plt.ylabel("# SOR Iterations")
    plt.title(
        rf"# iterations needed in SOR vs $\omega$ for 100x100 grid, 50 grow steps, $\eta = {eta}$"
    )
    plt.grid(True)

    if save:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
