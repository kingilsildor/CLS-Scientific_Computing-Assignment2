import numpy as np

from modules.config import NOISE_VALUE


def initialize_grid(N: int, value: int | float = 0.0) -> np.ndarray:
    """
    Initialize a grid of size N x N with a given value

    Params:
    -------
    - N (int): size of the grid
    - value (int | float): value to fill the grid with. Default is 0.0

    Returns:
    --------
    - grid (np.ndarray): grid of size N x N
    """
    if isinstance(N, int):
        value = float(value)
    if not isinstance(value, float):
        raise ValueError("Value should be an integer or a float")

    grid = np.full((N, N), fill_value=value, dtype=float)
    assert grid.shape == (N, N)
    return grid


def fill_center(grid: np.ndarray, value: float = 1.0) -> np.ndarray:
    """
    Fill the center of the grid with a given value

    Params:
    -------
    - grid (np.ndarray): grid to fill
    - value (float): value to fill the center with. Default is 1.0

    Returns:
    --------
    - grid (np.ndarray): grid with the center filled with the given value
    """
    N = grid.shape[0]
    x, y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

    mask = (0.4 < x) & (x < 0.6) & (0.4 < y) & (y < 0.6)
    grid[mask] = value

    assert grid.shape == (N, N)
    assert np.all(grid[mask] == value)
    return grid


def fill_noise(grid: np.ndarray) -> np.ndarray:
    """
    Add some noise to the grid

    Params:
    -------
    - grid (np.ndarray): grid to add noise to

    Returns:
    --------
    - grid (np.ndarray): grid with noise added
    """
    noise = np.random.normal(0, NOISE_VALUE, grid.shape)
    grid += noise
    return grid
