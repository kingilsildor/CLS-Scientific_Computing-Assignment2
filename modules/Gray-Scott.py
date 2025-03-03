import numpy as np
import matplotlib.pyplot as plt
from grid import initialize_grid, fill_center
from matplotlib.animation import FuncAnimation

DIRICHLET_VALUE = 1


def laplacian(chemical: np.ndarray, dx: float):
    """
    Calculate the laplacian of a 2D grid

    Params:
    -------
    - chemical (np.ndarray): 2D grid

    Returns:
    --------
    - laplacian (np.ndarray): laplacian of the input grid
    """
    laplacian = chemical.copy()
    laplacian[1:-1, 1:-1] = (
        chemical[:-2, 1:-1]
        + chemical[2:, 1:-1]
        + chemical[1:-1, :-2]
        + chemical[1:-1, 2:]
        - 4 * chemical[1:-1, 1:-1]
    ) * dx**2

    assert laplacian.shape == chemical.shape
    return laplacian


def gray_scott(
    U: np.ndarray,
    V: np.ndarray,
    Du: float,
    Dv: float,
    F: float,
    k: float,
    dt: float,
    dx: float,
):
    """
    Calculate the Gray-Scott reaction-diffusion model

    Params:
    -------
    - u (np.ndarray): concentration of the u chemical
    - v (np.ndarray): concentration of the v chemical
    - Du (float): diffusion rate of the u chemical
    - Dv (float): diffusion rate of the v chemical
    - F (float): feed rate
    - k (float): kill rate
    - dt (float): time step

    Returns:
    --------
    - u (np.ndarray): updated concentration of the u chemical
    - v (np.ndarray): updated concentration of the v chemical
    """
    u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]

    Lu = laplacian(u, dx)
    Lv = laplacian(v, dx)

    uvv = u * v * v

    u += (Du * Lu - uvv + F * (1 - u)) * dt
    v += (Dv * Lv + uvv - (F + k) * v) * dt

    u = boundary_conditions(u)
    v = boundary_conditions(v)

    assert u.shape == v.shape
    return u, v


def simulate_gray_scott(
    N: int,
    Du: float,
    Dv: float,
    F: float,
    k: float,
    dt: float,
    dx: float,
    steps: int,
):
    """
    Simulate the Gray-Scott reaction-diffusion model

    Params:
    -------
    - N (int): size of the grid
    - Du (float): diffusion rate of the u chemical
    - Dv (float): diffusion rate of the v chemical
    - F (float): feed rate
    - k (float): kill rate
    - dt (float): time step
    - steps (int): number of steps to simulate

    Returns:
    --------
    - u (np.ndarray): final concentration of the u chemical
    - v (np.ndarray): final concentration of the v chemical
    """
    U = initialize_grid(N, 1.0)
    V = fill_center(initialize_grid(N), 0.25)

    U = initialize_grid(N, 1.0)
    V = fill_center(initialize_grid(N), 0.25)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(U, cmap='viridis', vmin=0, vmax=1)
    im2 = axes[1].imshow(V, cmap='magma', vmin=0, vmax=1)
    axes[0].set_title("U Concentration")
    axes[1].set_title("V Concentration")

    def update(frame, U, V):
        U, V = gray_scott(U, V, Du, Dv, F, k, dt, dx)
        im1.set_data(U)
        im2.set_data(V)
        return [im1, im2]

    ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=True, fargs=(U, V))
    video_path = "results/gray_scott.mp4"
    ani.save(video_path, writer="ffmpeg", fps=30)

    plt.close(fig)


def boundary_conditions(grid: np.ndarray, condition: str = "periodic"):
    """
    Apply boundary conditions to a grid

    Params:
    -------
    - grid (np.ndarray): 2D grid
    - condition (str): boundary condition to apply. Default is "periodic"

    Returns:
    --------
    - grid (np.ndarray): grid with boundary conditions applied
    """
    assert condition in ["dirichlet", "neumann", "periodic"]
    new_grid = grid.copy()

    if condition == "dirichlet":
        new_grid[0, :] = DIRICHLET_VALUE
        new_grid[-1, :] = DIRICHLET_VALUE
        new_grid[:, 0] = DIRICHLET_VALUE
        new_grid[:, -1] = DIRICHLET_VALUE
    elif condition == "neumann":
        new_grid[0, :] = new_grid[1, :]
        new_grid[-1, :] = new_grid[-2, :]
        new_grid[:, 0] = new_grid[:, 1]
        new_grid[:, -1] = new_grid[:, -2]
    elif condition == "periodic":
        new_grid[0, :] = new_grid[-2, :]
        new_grid[-1, :] = new_grid[1, :]
        new_grid[:, 0] = new_grid[:, -2]
        new_grid[:, -1] = new_grid[:, 1]

    assert new_grid.shape == grid.shape
    return new_grid

def main():
    N = 100
    Du = 0.16
    Dv = 0.08
    F = 0.035
    k = 0.06
    dt = 1
    dx = 1
    steps = 30_000

    u, v = simulate_gray_scott(N, Du, Dv, F, k, dt, dx, steps)


if __name__ == "__main__":
    main()
