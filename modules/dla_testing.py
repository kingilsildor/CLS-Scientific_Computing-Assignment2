import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def successive_over_relaxation(
    grid, cluster, omega=1.8, epsilon=1e-5, max_iterations=5000
):
    """
    Run the Successive Under Relaxation method to solve the time independent diffusion equation

    Params: TODO
    -------
    - grid (np.ndarray): The initial spatial grid
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


def run_dla_simulation(grid, cluster, boundary, num_iterations):
    N = grid.shape[0]

    results_animation = []
    clusters = []
    clusters.append(cluster.copy())

    for _ in tqdm(range(num_iterations), desc="Iteration"):
        results, _, _ = successive_over_relaxation(
            grid, cluster, omega=1.8, epsilon=1e-5, max_iterations=1000
        )
        grid = results[-1]

        # Save grids for animations
        results_animation.append(grid.copy())

        # Get the probabilities of boundary cells
        boundary_coords = [
            coords for coords in boundary
        ]  # list of boundary coordinates
        boundary_concentration = [
            grid[coords] if grid[coords] >= 0 else 0 for coords in boundary_coords
        ]
        total_boundary_concentration = sum(boundary_concentration)

        # In case there is no more concentration, stop
        if np.abs(total_boundary_concentration) < 1e-10:
            break

        # Pick one cell randomly
        new_cell = np.random.choice(
            [i for i in range(len(boundary))],
            size=1,
            p=boundary_concentration / total_boundary_concentration,
        )[0]
        new_coords = boundary_coords[new_cell]

        # print(f'New cell to be added to the cluster: {new_coords}')

        # Add the new cell to the cluster
        cluster.add(new_coords)
        boundary.remove(new_coords)

        # Set the concentration of the cluster to 0 in the grid:
        grid[new_coords] = 0

        # Update the boundary
        # Left
        if new_coords[1] > 0 and (new_coords[0], new_coords[1] - 1) not in cluster:
            boundary.add((new_coords[0], new_coords[1] - 1))
        # Right
        if new_coords[1] < N - 1 and (new_coords[0], new_coords[1] + 1) not in cluster:
            boundary.add((new_coords[0], new_coords[1] + 1))
        # Top
        if new_coords[0] > 0 and (new_coords[0] - 1, new_coords[1]) not in cluster:
            boundary.add((new_coords[0] - 1, new_coords[1]))
        # Bottom
        if new_coords[0] < N - 1 and (new_coords[0] + 1, new_coords[1]) not in cluster:
            boundary.add((new_coords[0] + 1, new_coords[1]))

        clusters.append(cluster.copy())

    return results_animation, clusters


def animate_diffusion(
    results: np.ndarray,
    clusters: list,
    steps: int | None = None,
    save_animation: bool = False,
    animation_name: str = "dla.mp4",
) -> HTML:
    """
    Animate the diffusion of heat in a grid

    Params
    ------
    - results (np.ndarray): Grid with heat diffusion over time. Default: None
    - steps (int): Number of time steps. Default: None
    - save_animation (bool): Save the animation as a video file. Default: False
    - animation_name (str): Name of the animation file. Default: "diffusion.mp4"

    Returns
    -------
    - HTML: Animation of the diffusion of heat in a grid
    - Video file with the animation of the diffusion of heat in a grid
    """
    assert isinstance(results, np.ndarray), "Invalid results type, should be np.ndarray"
    steps = results.shape[0]

    fig, ax = plt.subplots()
    init_grid = results[0]
    im = ax.imshow(init_grid, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)

    x_points = [coords[1] for coords in clusters[0]]
    y_points = [coords[0] for coords in clusters[0]]
    scatter = ax.scatter(x_points, y_points, color="black", s=5)

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
