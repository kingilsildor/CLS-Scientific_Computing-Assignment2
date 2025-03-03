import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from PIL import Image
from tqdm import tqdm

from modules.config import DIRICHLET_VALUE
from modules.grid import fill_center, initialize_grid, fill_noise


@njit
def boundary_conditions(chemical: np.ndarray, boundary: str) -> np.ndarray:
    """
    Apply boundary conditions to a 2D grid

    Params:
    -------
    - chemical (np.ndarray): 2D grid
    - boundary (str): type of boundary condition to apply

    Returns:
    --------
    - chemical (np.ndarray): 2D grid with boundary conditions applied
    """
    if boundary == "neumann":
        chemical[0, :] = chemical[1, :]
        chemical[-1, :] = chemical[-2, :]
        chemical[:, 0] = chemical[:, 1]
        chemical[:, -1] = chemical[:, -2]
    elif boundary == "periodic":
        chemical[0, :] = chemical[-2, :]
        chemical[-1, :] = chemical[1, :]
        chemical[:, 0] = chemical[:, -2]
        chemical[:, -1] = chemical[:, 1]
    elif boundary == "dirichlet":
        chemical[0, :] = DIRICHLET_VALUE
        chemical[-1, :] = DIRICHLET_VALUE
        chemical[:, 0] = DIRICHLET_VALUE
        chemical[:, -1] = DIRICHLET_VALUE
    else:
        raise ValueError(f"Invalid boundary condition: {boundary}")
    return chemical


@njit
def laplacian(chemical: np.ndarray) -> np.ndarray:
    """
    Calculate the laplacian of a 2D grid

    Params:
    -------
    - chemical (np.ndarray): 2D grid

    Returns:
    --------
    - laplacian (np.ndarray): laplacian of the input grid
    """
    laplacian = np.zeros_like(chemical)

    laplacian[1:-1, 1:-1] = (
        chemical[:-2, 1:-1]
        + chemical[2:, 1:-1]
        + chemical[1:-1, :-2]
        + chemical[1:-1, 2:]
        - 4 * chemical[1:-1, 1:-1]
    )

    laplacian = boundary_conditions(laplacian, "neumann")

    assert laplacian.shape == chemical.shape
    return laplacian


@njit
def gray_scott(
    u: np.ndarray,
    v: np.ndarray,
    Du: float,
    Dv: float,
    F: float,
    k: float,
    dx: float,
    dt: float,
    boundary: str,
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
    - dx (float): spatial step
    - dt (float): time step
    - boundary (str): type of boundary condition to apply

    Returns:
    --------
    - u (np.ndarray): updated concentration of the u chemical
    - v (np.ndarray): updated concentration of the v chemical
    """
    Lu = laplacian(u)
    Lv = laplacian(v)

    uvv = u * v * v

    u += (Du * Lu * 1 / dx**2 - uvv + F * (1 - u)) * dt
    v += (Dv * Lv * 1 / dx**2 + uvv - (F + k) * v) * dt

    assert u.shape == v.shape
    return u, v


def simulate_gray_scott(
    N: int,
    Du: float,
    Dv: float,
    F: float,
    k: float,
    dx: float,
    dt: float,
    steps: int,
    noise_u: bool = False,
    noise_v: bool = False,
    boundary: str = "neumann",
    chemical: str = "v",
    info: bool = False,
):
    """
    Simulate the Gray-Scott reaction-diffusion model with side-by-side animation.

    Params:
    -------
    - N (int): size of the grid
    - Du (float): diffusion rate of the u chemical
    - Dv (float): diffusion rate of the v chemical
    - F (float): feed rate
    - k (float): kill rate
    - dx (float): spatial step
    - dt (float): time step
    - steps (int): number of steps to simulate
    - noise_u (bool): whether to add noise to the u chemical
    - noise_v (bool): whether to add noise to the v chemical
    - boundary (str): type of boundary condition to apply (neumann, periodic, dirichlet)
    - chemical (str): chemical to visualize (u or v)
    - info (bool): whether to display simulation info

    Returns:
    --------
    - u (np.ndarray): final concentration of the u chemical
    - v (np.ndarray): final concentration of the v chemical
    """
    u = initialize_grid(N, 1.0)
    v = fill_center(initialize_grid(N))
    if noise_u:
        u = fill_noise(u)
    if noise_v:
        v = fill_noise(v)
    assert isinstance(u, np.ndarray) & isinstance(v, np.ndarray)
    assert u.shape == v.shape

    colour = "viridis"

    for step in tqdm(range(steps), desc="Gray-Scott simulation"):
        u, v = gray_scott(u, v, Du, Dv, F, k, dx, dt, boundary)

        if step % 100 == 0:
            filename = "results/gs_{:02d}.png".format(step // 100)
            if chemical == "u":
                plt.imshow(u, cmap=colour)
            elif chemical == "v":
                plt.imshow(v, cmap=colour)
            else:
                raise ValueError(f"Invalid chemical: {chemical}")

            if info:
                plt.title(
                    f"Gray-Scott Step {step} for chemical ${chemical}$\n $D_u={Du}$, $D_v={Dv}$, $F={F}$, $k={k}$, $\\delta x={dx}$, $\\delta t={dt}$\n {boundary} boundary"
                )
            else:
                plt.title(f"Gray-Scott Step {step} for chemical ${chemical}$\n")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(filename)
    plt.close()


def create_gif(
    image_folder: str,
    output_gif: str,
    duration: int = 100,
    loop: int = 0,
    delete_images: bool = True,
):
    """
    Create a GIF from multiple PNG images.

    Params:
    -------
    - image_folder (str): Path to the folder containing PNG images.
    - output_gif (str): Path to save the output GIF.
    - duration (int): Duration of each frame in milliseconds (default 100ms).
    - loop (int): Number of times the GIF loops (0 = infinite loop).
    - delete_images (bool): Whether to delete the PNG images after creating the GIF.
    """
    images = sorted(glob.glob(f"{image_folder}/*.png"))
    assert images, f"No images found in {image_folder}"

    frames = [Image.open(img) for img in images]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )

    if delete_images:
        for img in images:
            os.remove(img)
