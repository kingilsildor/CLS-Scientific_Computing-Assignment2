import time
from typing import Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from modules.DLA_model import Diffusion


def successive_over_relaxation(
    grid: np.ndarray,
    cluster: Set[Tuple[int, int]],
    omega: float | None = 1.8,
    epsilon: float | None = 1e-5,
    max_iterations: int | None = 5000,
) -> Tuple[list, list, int]:
    """
    Run the Successive Under Relaxation method to solve the time independent diffusion equation

    Params:
    -------
    - grid (np.ndarray): The initial spatial grid
    - cluster (Set[Tuple[int, int]]): The coordinates of the cluster
    - epsilon (float, optional): The convergence criterion. Defaults to 1e-5.
    - max_iterations (int, optional): The maximum number of iterations. Defaults to 100000.
    - omega (float, optional): The relaxation factor. Defaults to 1.8

    Returns:
    --------
    - results (List[np.ndarray]): A list of spatial grids at each iteration
    - residuals (List[float]): The residuals at each iteration
    - k (int): The number of iterations
    """
    residuals = []
    results = []
    results.append(grid.copy())
    N = grid.shape[0] - 1

    delta = float("inf")
    k = 0

    while delta > epsilon and k < max_iterations:
        delta = 0

        # First column
        for i in range(1, N):
            if (i, 0) in cluster:
                continue

            old_cell = grid[i][0].copy()
            grid[i][0] = (
                omega
                / 4
                * (grid[i + 1][0] + grid[i - 1][0] + grid[i][1] + grid[i][N - 1])
                + (1 - omega) * grid[i][0]
            )

            if np.abs(grid[i][0] - old_cell) > delta:
                delta = np.abs(grid[i][0] - old_cell)

        for j in range(1, N):
            for i in range(1, N):
                if (i, j) in cluster:
                    continue

                old_cell = grid[i][j].copy()
                grid[i][j] = (
                    omega
                    / 4
                    * (
                        grid[i + 1][j]
                        + grid[i - 1][j]
                        + grid[i][j + 1]
                        + grid[i][j - 1]
                    )
                    + (1 - omega) * grid[i][j]
                )

                delta = max(delta, np.abs(grid[i][j] - old_cell))

        # Last column
        for i in range(1, N):
            if (i, N) in cluster:
                continue

            old_cell = grid[i][N].copy()
            grid[i][N] = (
                omega
                / 4
                * (grid[i + 1][N] + grid[i - 1][N] + grid[i][0] + grid[i][N - 1])
                + (1 - omega) * grid[i][N]
            )

            if np.abs(grid[i][N] - old_cell) > delta:
                delta = np.abs(grid[i][N] - old_cell)

        results.append(grid.copy())
        residuals.append(delta)

        k += 1

    return results, residuals, k


def run_dla_simulation(
    diffusion: Diffusion, eta: float | None = 1, num_iterations: int | None = 100
) -> Tuple[list, list]:
    """
    Run the DLA simulation with the Successive Over Relaxation method

    Params:
    -------
    - diffusion (Diffusion): The diffusion object
    - num_iterations (int, optional): The number of iterations to run the simulation for. Defaults to 100.

    Returns:
    --------
    - results_animation (List[np.ndarray]): A list of the grids at each iteration
    - clusters (List[Set[Tuple[int, int]]]): A list of the cluster at each iteration
    """
    assert eta >= 0

    results_animation = []
    clusters = []
    clusters.append(diffusion.cluster.copy())

    N = diffusion.N
    diffusion.grid = np.array(
        [[1 - row / (N - 1) for col in range(N)] for row in range(N)]
    )  # Initial guess

    # Set the initial cluster again
    for coords in diffusion.cluster:
        diffusion.grid[coords] = 0

    for i in tqdm(range(num_iterations), desc="Iteration"):
        print(f"Iteration {i}")
        start_time_full_loop = time.time()

        results, _, _ = successive_over_relaxation(
            diffusion.grid, diffusion.cluster, epsilon=1e-5
        )
        diffusion.grid = results[-1]

        print("--- %s seconds SOR ---" % (time.time() - start_time_full_loop))

        start_time_full_loop = time.time()

        # Save grids for animations
        results_animation.append(diffusion.grid.copy())

        # Get the probabilities of boundary cells
        boundary_coords = [
            coords for coords in diffusion.perimeter
        ]  # list of boundary coordinates

        print(
            "--- %s seconds append copy + get coords ---"
            % (time.time() - start_time_full_loop)
        )

        start_time_full_loop = time.time()

        boundary_concentration = [
            diffusion.grid[coords] ** eta if diffusion.grid[coords] >= 0 else 0
            for coords in boundary_coords
        ]
        total_boundary_concentration = sum(boundary_concentration)

        print(
            "--- %s seconds get probabilities and sum ---"
            % (time.time() - start_time_full_loop)
        )

        start_time_full_loop = time.time()

        # In case there is no more concentration, stop
        if np.abs(total_boundary_concentration) < 1e-10:
            print(
                f"The simulation has stopped after {i} steps because there is no more concentration"
            )
            break

        # Pick one cell randomly
        new_cell_idx = np.random.choice(
            [i for i in range(len(diffusion.perimeter))],
            size=1,
            p=boundary_concentration / total_boundary_concentration,
        )[0]
        new_coords = boundary_coords[new_cell_idx]

        print("--- %s seconds pick new cell---" % (time.time() - start_time_full_loop))

        start_time_full_loop = time.time()

        # Add the new cell to the cluster
        diffusion.add_to_cluster(new_coords)

        print("--- %s seconds add to cluster---" % (time.time() - start_time_full_loop))

        start_time_full_loop = time.time()

        # Periodic conditions
        if new_coords[1] == 0:
            diffusion.add_to_cluster((new_coords[0], diffusion.N - 1))
        elif new_coords[1] == diffusion.N - 1:
            diffusion.add_to_cluster((new_coords[0], 0))

        print("--- %s seconds add boundary ---" % (time.time() - start_time_full_loop))

        start_time_full_loop = time.time()

        clusters.append(diffusion.cluster.copy())
        print("--- %s seconds full loop ---" % (time.time() - start_time_full_loop))

    return results_animation, clusters


def animate_diffusion(
    results: list,
    clusters: list,
    save_animation: bool = False,
    animation_name: str = "dla.mp4",
) -> HTML:
    """
    Animate the diffusion of heat in a grid

    Params
    ------
    - results (list): List of grids after each growth step
    - clusters (list): List of clusters after each growth step
    - save_animation (bool): Save the animation as a video file. Default: False
    - animation_name (str): Name of the animation file. Default: "diffusion.mp4"

    Returns
    -------
    - HTML: Animation of the diffusion of heat in a grid
    - Video file with the animation of the diffusion of heat in a grid
    """
    steps = len(results)

    fig, ax = plt.subplots()
    init_grid = results[0]
    im = ax.imshow(init_grid, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)

    x_points = [coords[1] for coords in clusters[0]]
    y_points = [coords[0] for coords in clusters[0]]
    scatter = ax.scatter(x_points, y_points, color="black", s=2)

    def _animate(frame):
        im.set_array(results[frame])
        ax.set_title(f"Time step {frame}")
        x_points = [coords[1] for coords in clusters[frame]]
        y_points = [coords[0] for coords in clusters[frame]]
        scatter.set_offsets(np.column_stack((x_points, y_points)))

        return im, scatter

    ani = FuncAnimation(fig, _animate, frames=steps, interval=100, blit=False)
    if save_animation:
        ani.save(
            animation_name,
            writer="ffmpeg",
            fps=60,
        )
    # plt.cla()
    plt.close(fig)

    return HTML(ani.to_html5_video())
