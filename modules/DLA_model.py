import numpy as np
from typing import Tuple, Dict, Set

CLUSTER_VALUE = 0


class Diffusion:
    def __init__(self, N: int, initial_point: str = "bottom"):
        """
        Params:
        -------
        - N: int, size of the grid
        - initial_point: str, where to start the cluster
        """
        self.N = N
        self.grid = np.zeros((N, N))
        assert self.grid.shape == (N, N)

        if initial_point == "bottom":
            coords = (0, N // 2)
            self.grid[coords] = CLUSTER_VALUE
        elif initial_point == "top":
            coords = (N - 1, N // 2)
            self.grid[coords] = CLUSTER_VALUE
        elif initial_point == "center":
            coords = (N // 2, N // 2)
            self.grid[coords] = CLUSTER_VALUE
        else:
            raise ValueError(
                "Invalid initial point, choose from 'bottom', 'top' or 'center'"
            )

        assert np.sum(self.grid) == CLUSTER_VALUE

        self.cluster: Set[Tuple[int, int]] = set()
        self.perimeter: Set[Tuple[int, int]] = set()

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
        neighbours.add((x - 1, y))
        neighbours.add((x + 1, y))
        neighbours.add((x, y - 1))
        neighbours.add((x, y + 1))

        assert len(neighbours) == 4
        assert coords not in neighbours
        assert isinstance(neighbours, set)
        return neighbours


def main():
    diffusion = Diffusion(5, initial_point="center")
    diffusion.add_to_cluster((2, 2))
    diffusion.add_to_cluster((2, 3))


if __name__ == "__main__":
    main()
