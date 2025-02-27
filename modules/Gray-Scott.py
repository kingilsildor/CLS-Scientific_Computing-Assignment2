import numpy as np
import matplotlib.pyplot as plt
from grid import initialize_grid, fill_center
from matplotlib.animation import FuncAnimation


def laplacian(chemical: np.ndarray):
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
    )

    assert laplacian.shape == chemical.shape
    return laplacian


def gray_scott(
    u: np.ndarray, v: np.ndarray, Du: float, Dv: float, F: float, k: float, dt: float
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
    Lu = laplacian(u)
    Lv = laplacian(v)

    uvv = u * v * v

    u += (Du * Lu - uvv + F * (1 - u)) * dt
    v += (Dv * Lv + uvv - (F + k) * v) * dt

    u = boundary_conditions(u)
    v = boundary_conditions(v)

    assert u.shape == v.shape
    return u, v


def simulate_gray_scott(
    N: int, Du: float, Dv: float, F: float, k: float, dt: float, steps: int
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
    u = initialize_grid(N, 1.0)
    v = fill_center(initialize_grid(N), 0.25)

    fig, ax = plt.subplots()
    im = ax.imshow(v, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Gray-Scott Reaction-Diffusion (v)")
    plt.colorbar(im)

    # Update function for animation
    def update(frame, u, v):
        u, v = gray_scott(u, v, Du, Dv, F, k, dt)
        im.set_data(v)
        return [im]

    ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True, fargs=(u, v))
    plt.show()


def boundary_conditions(u: np.ndarray):
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]

    return u


def main():
    N = 100
    Du = 0.16
    Dv = 0.08
    F = 0.035
    k = 0.06
    dt = 1
    steps = 1000

    u, v = simulate_gray_scott(N, Du, Dv, F, k, dt, steps)


if __name__ == "__main__":
    main()
