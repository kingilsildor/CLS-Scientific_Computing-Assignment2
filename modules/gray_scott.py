import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from PIL import Image
from tqdm import tqdm

from modules.config import DIRICHLET_VALUE, DPI, FIG_SIZE, PRINT_EVERY
from modules.grid import fill_center, fill_noise, initialize_grid


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
def laplacian(chemical: np.ndarray, boundary: str) -> np.ndarray:
    """
    Calculate the laplacian of a 2D grid

    Params:
    -------
    - chemical (np.ndarray): 2D grid
    - boundary (str): type of boundary condition to apply

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

    laplacian = boundary_conditions(laplacian, boundary)

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
) -> tuple:
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
    Lu = laplacian(u, boundary)
    Lv = laplacian(v, boundary)

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
) -> None:
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
        u, v = gray_scott(
            u=u, v=v, Du=Du, Dv=Dv, F=F, k=k, dx=dx, dt=dt, boundary=boundary
        )
        noise = noise_u or noise_v

        if step % PRINT_EVERY == 0:
            plt.figure(figsize=FIG_SIZE, dpi=DPI)
            filename = f"results/gs_{((step // 100) + 1):02d}_chemical={chemical}_D_u={Du}_D_v={Dv}_F={F}_k={k}_noise={noise}.png"
            if chemical == "u":
                im = plt.imshow(u, cmap=colour)
            elif chemical == "v":
                im = plt.imshow(v, cmap=colour)
            else:
                raise ValueError(f"Invalid chemical: {chemical}")

            # Enable the following line to display the colorbar
            # if step == steps - PRINT_EVERY:
            #     plt.colorbar(im)
            if info:
                plt.title(
                    f"Gray-Scott Step {step + PRINT_EVERY} for chemical ${chemical}$\n $D_u={Du}$, $D_v={Dv}$, $F={F}$, $k={k}$, $\\delta x={dx}$, $\\delta t={dt}$\n {boundary} boundary"
                )
            else:
                plt.title(
                    f"Gray-Scott Step {step + PRINT_EVERY} for chemical ${chemical}$\n"
                )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()


def get_images(image_folder: str) -> list:
    """
    Get a list of PNG images from a folder.

    Params:
    -------
    - image_folder (str): Path to the folder containing PNG images.

    Returns:
    --------
    - images (list): List of PNG images in the folder.
    """
    images = sorted(glob.glob(f"{image_folder}/gs_[0-9][0-9]_*.png"))
    return images


def delete_images(image_folder: str) -> None:
    """
    Delete all PNG images in a folder.

    Params:
    -------
    - image_folder (str): Path to the folder containing PNG images.

    Returns:
    --------
    - None: If no images are found in the folder.
    """
    images = get_images(image_folder)
    if not images:
        return

    for img in images:
        os.remove(img)
    print(f"Deleted {len(images)} images from {image_folder}")


def create_gif(
    image_folder: str,
    output_gif: str,
    duration: int = 100,
    loop: int = 0,
) -> None:
    """
    Create a GIF from multiple PNG images.

    Params:
    -------
    - image_folder (str): Path to the folder containing PNG images.
    - output_gif (str): Path to save the output GIF.
    - duration (int): Duration of each frame in milliseconds (default 100ms).
    - loop (int): Number of times the GIF loops (0 = infinite loop).
    - delete_images (bool): Whether to delete the PNG images after creating the GIF.

    Returns:
    --------
    - None: If no images are found in the folder/
    """
    images = get_images(image_folder)
    if not images:
        return

    frames = [Image.open(img) for img in images]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )


if __name__ == "__main__":
    delete_images("results")
    N = 100
    steps = 5_000
    dx = 1
    dt = 1

    reaction_diffusion_params = {
        "snowflake": {"Du": 0.16, "Dv": 0.08, "F": 0.035, "k": 0.06},
        "spots": {"Du": 0.1, "Dv": 0.05, "F": 0.035, "k": 0.065},
        "star": {"Du": 0.16, "Dv": 0.08, "F": 0.022, "k": 0.051},
        "wave": {"Du": 0.12, "Dv": 0.08, "F": 0.018, "k": 0.051},
    }
    for noise in [True, False]:
        for category, params in reaction_diffusion_params.items():
            Du = params["Du"]
            Dv = params["Dv"]
            F = params["F"]
            k = params["k"]
            simulate_gray_scott(
                N,
                Du,
                Dv,
                F,
                k,
                dx,
                dt,
                steps,
                chemical="v",
                boundary="neumann",
                info=False,
                noise_u=noise,
                noise_v=False,
            )
            # Run the following line to create a GIF
            # create_gif(
            #     "results", f"results/gray_scott_{category}_{noise}.gif", duration=100
            # )
            # print(f"Created GIF for {category} reaction-diffusion pattern")
            # delete_images("results")
