import matplotlib.pyplot as plt
import numpy as np


def create_grid(N: int) -> np.ndarray:
    """
    Create a grid of size N x N with zeros.

    Params
    ------
    - N (int): size of the grid

    Returns
    -------
    - grid (np.ndarray): grid of size N x N with zeros
    """
    grid = np.zeros((N, N))
    assert grid.shape == (N, N)
    return grid


def init_grid(grid: np.ndarray, initial_point: str = "bottom") -> np.ndarray:
    """
    Initialize the grid with a single particle at the center of the grid.

    Params
    ------
    - grid (np.ndarray): grid of size N x N
    - initial_point (str): initial point of the particle, can be "bottom", "center" or "top". Default is "bottom"

    Returns
    -------
    - grid (np.ndarray): grid of size N x N with a single particle at the initial point
    """
    grid = grid.copy()
    assert grid.shape[0] == grid.shape[1]
    N = grid.shape[0]

    if initial_point == "bottom":
        grid[N - 1, N // 2] = 2
    elif initial_point == "center":
        grid[N // 2, N // 2] = 2
    elif initial_point == "top":
        grid[0, N // 2] = 2
    else:
        ValueError("Invalid initial point. Must be 'bottom', 'center' or 'top'.")

    assert grid.shape == (N, N)
    assert np.sum(grid == 2) == 1
    return grid


def plot_grid(grid: np.ndarray):
    plt.imshow(grid, interpolation="nearest")
    plt.show()


if __name__ == "__main__":
    N = 100
    grid = create_grid(N)
    grid = init_grid(grid)
    plot_grid(grid)
