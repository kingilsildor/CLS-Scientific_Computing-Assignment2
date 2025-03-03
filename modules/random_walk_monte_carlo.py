import random
from typing import Set, Tuple

import matplotlib.pyplot as plt
from config import CLUSTER_VALUE, CONCENTRATION_VALUE
from grid import initialize_grid

N = 100


class RandomWalker:
    def __init__(self, N: int, initial_point: str = "bottom"):
        self.grid = initialize_grid(N)
        self.N = N
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
        return [0, random.randint(0, N - 1)]

    def add_to_cluster(self, coords: Tuple[int, int]):
        """Adds a position to the cluster and updates the perimeter."""
        self.cluster.add(coords)
        self.grid[coords[0], coords[1]] = CLUSTER_VALUE
        self.perimeter.update(self.get_neighbours(coords))

    def get_neighbours(self, coords: Tuple[int, int]):
        """Returns neighboring positions of a given cell."""
        x, y = coords
        neighbors = set()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.N and 0 <= ny < self.N:
                neighbors.add((nx, ny))
        return neighbors

    def move_walker(self):
        """Moves the walker randomly and checks if it hits the cluster."""
        directions = ["up", "down", "left", "right"]
        row, col = self.walker

        # Pick a random movement direction
        move = random.choice(directions)
        if move == "up":
            row -= 1
        elif move == "down":
            row += 1
        elif move == "left":
            col -= 1
        elif move == "right":
            col += 1

        # Periodic boundary conditions (left/right wrapping)
        if col < 0:
            col = self.N - 1
        elif col >= self.N:
            col = 0

        # Check if the walker touches the cluster
        if any(
            (row + dx, col + dy) in self.cluster
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ):
            self.add_to_cluster((row, col))  # Convert walker into cluster point
            self.walker = self.initialize_random_walker()  # Spawn new walker
        elif 0 <= row < self.N:  # Walker stays within bounds
            self.walker = [row, col]
        else:  # Walker falls out of top/bottom → Restart new walker
            self.walker = self.initialize_random_walker()

    def run_simulation(self, steps: int = 1000):
        """Runs the random walker simulation for a given number of steps."""
        for _ in range(steps):
            self.move_walker()

        # Plot only the final state
        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid, cmap="coolwarm", origin="upper")  # Show final grid
        plt.colorbar(label="Grid State")  # Show legend
        plt.title("Final Diffusion-Limited Aggregation Grid")
        plt.show()


simulation = RandomWalker(N=100, initial_point="bottom")
simulation.run_simulation(steps=5000000)
