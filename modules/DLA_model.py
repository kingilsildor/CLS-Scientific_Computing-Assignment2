import numpy as np
from typing import Tuple, List, Set

CLUSTER_VALUE = 0

class Diffusion:
    def __init__(self, N: int, initial_point: str = 'bottom'):
        self.N = N
        self.grid = np.zeros((N, N))
        assert self.grid.shape == (N, N)

        if initial_point == 'bottom':
            coords = (0, N // 2)
            self.grid[coords] = CLUSTER_VALUE
        elif initial_point == 'top':
            coords = (N - 1, N // 2)
            self.grid[coords] = CLUSTER_VALUE
        elif initial_point == 'center':
            coords = (N // 2, N // 2)
            self.grid[coords] = CLUSTER_VALUE
        else:
            raise ValueError("Invalid initial point, choose from 'bottom', 'top' or 'center'")
        
        assert np.sum(self.grid) == CLUSTER_VALUE

        self.cluster: Set[Tuple[int, int]] = set()

    def add_to_cluster(self, coords: Tuple[int, int]):
        self.cluster.add(coords)
    

