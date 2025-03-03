from typing import Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Grid:
    def __init__(self, N: int, eta: int = 1, initial_point: str = "bottom"):
        """
        Initialize the grid with a single particle at the center of the grid.

        Params
        ------
        - N (int): size of the grid
        - initial_point (str): initial point of the particle, can be "bottom", "center" or "top". Default is "bottom"
        """
        self.cluster: Set[Tuple[int, int]] = set()
        self.perimeter: Set[Tuple[int, int]] = set()

        self.N = N
        grid = np.zeros((N, N))
        assert grid.shape == (N, N), (
            f"Grid shape is {grid.shape} but should be {(N, N)}"
        )
        self.grid = grid
        self.initial_point = initial_point
        self.initialize_seed()

        self.eta = eta

    def initialize_seed(self):
        """
        Initialize the grid with a single particle at the center of the grid.
        """
        N = self.N

        if self.initial_point == "bottom":
            coords = (N - 1, N // 2)
            self.add_to_cluster(coords)
        elif self.initial_point == "center":
            coords = (N // 2, N // 2)
            self.add_to_cluster(coords)
        elif self.initial_point == "top":
            coords = (0, N // 2)
            self.add_to_cluster(coords)
        else:
            ValueError("Invalid initial point. Must be 'bottom', 'center' or 'top'.")

        assert coords in self.cluster, "Initial point not added to the cluster."

    def add_to_cluster(self, coord: Tuple[int, int]):
        """
        Add a particle to the cluster.

        Params
        ------
        - coord (Tuple[int, int]): coordinates of the particle to add

        Returns
        -------
        - None if the particle is already in the cluster or outside the grid
        """
        assert isinstance(coord, tuple), "coord must be a tuple."
        assert len(coord) == 2, "coord must have 2 elements."
        assert all(isinstance(_, int) for _ in coord), "All elements must be integers."

        x, y = coord
        N = self.N

        if coord in self.cluster:
            return
        if x < 0 or x >= N or y < 0 or y >= N:
            return

        self.cluster.add(coord)
        self.update_perimeter(coord)

    def get_neighbours(self, coord: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        Get the neighbours of a particle.

        Params
        ------
        - coord (Tuple[int, int]): coordinates of the particle

        Returns
        -------
        - Set[Tuple[int, int]]: set of neighbours of the particle
        """
        x, y = coord
        neighbours = {
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        }
        neighbours -= self.cluster
        assert coord not in neighbours, "Neighbours should not contain the particle."
        assert neighbours & self.cluster == set(), (
            "Neighbours should not intersect the cluster."
        )
        return neighbours

    def update_perimeter(self, coord: Tuple[int, int]):
        """
        Update the perimeter of the cluster.

        Params
        ------
        - coord (Tuple[int, int]): coordinates of the particle
        """
        neighbours = self.get_neighbours(coord)
        self.perimeter |= neighbours

    def get_probabilities(self, perimeter, c_function=None):
        if c_function is None:
            c_values = {candidate: 1 for candidate in perimeter}
        else:
            c_values = {candidate: c_function(*candidate) for candidate in perimeter}

        eta = self.eta
        denominator = sum(c_values[candidate] ** eta for candidate in perimeter)

        probabilities = [
            (c_values[candidate] ** eta) / denominator for candidate in perimeter
        ]

        return probabilities, c_values

    def get_next_particle(self, c_values) -> Tuple[int, int]:
        """
        Get the next particle to add to the cluster.

        Returns
        -------
        - Tuple[int, int]: coordinates of the next particle
        """
        perimeter = list(self.perimeter)
        probabilities, c_values = self.get_probabilities(perimeter, c_values)

        next_index = np.random.choice(len(perimeter), p=probabilities)
        next_particle = perimeter[next_index]
        return next_particle

    def grow_cluster(self, c_values):
        """
        Grow the cluster by adding particles to it.
        """
        next_particle = self.get_next_particle(c_values)
        self.add_to_cluster(next_particle)

    def simulate(self, c_values):
        """
        Simulate the growth of the cluster.
        """
        for _ in tqdm(range(10_000)):
            self.grow_cluster(c_values)

    def visualize_cluster(self):
        """
        Visualize the cluster.
        """
        cluster_points = list(self.cluster)
        if not cluster_points:
            print("No points in the cluster to visualize.")
            return

        x, y = zip(*cluster_points)
        self.grid[x, y] = 1
        plt.imshow(self.grid)
        plt.show()


if __name__ == "__main__":
    N = 100
    grid = Grid(N, initial_point="bottom")
    c_values = None
    grid.simulate(c_values)
    grid.visualize_cluster()
