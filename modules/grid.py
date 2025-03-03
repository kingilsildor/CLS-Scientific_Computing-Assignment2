import numpy as np

def initialize_grid(N: int) -> np.ndarray:
    """
    Initialize a grid of size N x N

    Params:
    -------
    - N: int, size of the grid

    Returns:
    --------
    - grid: np.ndarray, grid of size N x N
    """
    grid = np.zeros((N, N))
    assert grid.shape == (N, N)

    return grid