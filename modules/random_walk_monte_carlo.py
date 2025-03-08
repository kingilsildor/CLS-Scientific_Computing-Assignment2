import random
from typing import Set, Tuple

import matplotlib.pyplot as plt

from modules.config import CLUSTER_VALUE, CONCENTRATION_VALUE, DPI
from modules.grid import initialize_grid


class RandomWalker:
    def __init__(self, N: 100, p_stick: float, initial_point: str = "bottom"):
        self.grid = initialize_grid(N)
        self.N = N
        self.p_stick = p_stick
        self.cluster: Set[Tuple[int, int]] = set()
        self.perimeter: Set[Tuple[int, int]] = set()

        if initial_point == "top":
            coords = (0, N // 2)
            self.add_to_cluster(coords)
            self.grid[-1, :] = CONCENTRATION_VALUE
        elif initial_point == "bottom":
            coords = (N - 1, N // 2)
            self.add_to_cluster(coords)
            self.grid[0, :] = CONCENTRATION_VALUE
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

        assert self.grid[coords] == CLUSTER_VALUE
        assert self.cluster == {coords}
        assert self.perimeter == self.get_neighbours(coords)

        self.walker = self.initialize_random_walker()

    def initialize_random_walker(self):
        return [0, random.randint(0, self.N - 1)]

    def add_to_cluster(self, coords: Tuple[int, int]):
        """Adds a position to the cluster and updates the perimeter."""
        x, y = coords
        self.cluster.add(coords)
        self.grid[x, y] = CLUSTER_VALUE

        # Add neighbours to the perimeter
        neighbours = self.get_neighbours(coords)
        neighbours -= self.cluster

        # self.perimeter.update(self.get_neighbours(coords))
        self.perimeter.update(neighbours)
        self.perimeter.discard(coords)

        assert coords not in self.perimeter
        assert coords in self.cluster

    def get_neighbours(self, coords: Tuple[int, int]):
        """Returns neighbouring positions of a given cell."""
        x, y = coords
        neighbours = set()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.N and 0 <= ny < self.N:
                neighbours.add((nx, ny))

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

    def move_walker(self, p_stick: float = 0.1):
        """Moves the walker randomly and checks if it hits the cluster."""
        row, col = self.walker

        possible_moves = [
            (row + dx, col + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]

        valid_moves = [
            (new_row, (new_col + self.N) % self.N)
            for new_row, new_col in possible_moves
            if (0 <= new_row < self.N)
        ]

        if valid_moves:
            self.walker = random.choice(valid_moves)
        else:
            self.walker = self.initialize_random_walker()

        if any(
            (self.walker[0] + dx, self.walker[1] + dy) in self.cluster
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ):
            if random.random() < self.p_stick:
                self.add_to_cluster(tuple(self.walker))
                self.walker = self.initialize_random_walker()

    def run_simulation(self, steps: int = 1000):
        """Runs the random walker simulation for a given number of steps."""
        for _ in range(steps):
            self.move_walker()
            if len(self.cluster) == self.N * 2:
                # print("Cluster has reached the grid boundary.")
                break

    def plot(
        self,
        p_stick: float = 1,
        save: bool = False,
        filename: str = "random_walker.png",
    ):
        fig, ax = plt.subplots()
        im = ax.imshow(self.grid, cmap="Blues")
        plt.colorbar(im)

        x_points = [coords[1] for coords in self.cluster]
        y_points = [coords[0] for coords in self.cluster]
        ax.scatter(x_points, y_points, color="black", s=2)
        ax.set_title(
            rf"Random-Walker cluster for P_s = {p_stick}$ and {len(self.cluster) - 1} growth steps"
        )

        if save:
            plt.savefig(filename, dpi=DPI, bbox_inches="tight")

        plt.show()
        # # Plot only the final state
        # plt.figure(figsize=(6, 6))
        # plt.imshow(self.grid, cmap="coolwarm", origin="upper")  # Show final grid
        # plt.colorbar(label="Grid State")  # Show legend
        # plt.title("Final Diffusion-Limited Aggregation Grid")
        # plt.show()


# simulation = RandomWalker(N=100, initial_point="bottom")
# simulation.run_simulation(steps=10000000)
