import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import heapq
import random

from . import elements as el
from . import utils


class Simulation:
    """
    A class that simulates pedestrian movement in a grid environment.

    This class manages the simulation of pedestrians navigating a grid with obstacles
    and targets. It handles the initialization of the simulation environment, updates the
    positions of pedestrians based on certain rules, and computes distances to targets.
    The simulation can be configured to use different distance computation algorithms.

    Parameters:
    -----------
    config : el.SimulationConfig
        A configuration object that specifies the simulation parameters, including the
        grid size, initial positions of pedestrians, obstacles, targets, and the
        distance computation method.
    random_seed : int, optional
        The seed for the random number generator to ensure reproducibility of random operations.
        Default is 42.

    Attributes:
    -----------
    grid_size : el.GridSize
        The size of the grid (width and height) as defined in the configuration.
    width : int
        The width of the simulation grid.
    height : int
        The height of the simulation grid.
    grid : npt.NDArray[el.ScenarioElement]
        A 2D array representing the grid state, initialized with empty cells and populated
        with pedestrians, targets, and obstacles as specified in the configuration.
    pedestrians : list[el.Pedestrian]
        A list of pedestrian objects representing their initial states in the simulation.
    obstacles : list[el.Obstacle]
        A list of obstacle objects present in the simulation.
    targets : list[el.Target]
        A list of target objects that pedestrians are trying to reach.
    distance_computation : str
        The algorithm used for computing distances to targets (e.g., "naive" or "dijkstra").
    output_filename : str
        The filename where the simulation output will be saved.

    Methods:
    --------
    __init__(config: el.SimulationConfig, random_seed: int = 42):
        Initializes the Simulation class with the given configuration.
    update(perturb: bool = True) -> bool:
        Updates the positions of pedestrians based on their proximity to targets, returning
        whether all pedestrians have reached their targets.
    mp_reset_and_check_active():
        Resets attributes of measuring points and checks if they are active for the current step.
    mp_compute_density():
        Calculates pedestrian density in each measuring point area if it is active.
    mp_compute_speed():
        Computes the average speed of tracked pedestrians within active measuring points.
    compute_ped_distance(pedestrian, x_dif, y_dif):
        Updates pedestrian's distance based on movement in the x and y directions.
    print_mp_result():
        Prints the results for each measuring point at the end of its measuring period.
    get_next_move_position(pedestrian, distance_grid):
        Determines the next move position for a pedestrian based on the distance grid.
    get_grid() -> npt.NDArray[el.ScenarioElement]:
        Returns the current state of the grid.
    get_distance_grid() -> npt.NDArray[np.float64]:
        Returns a grid with distances to the nearest target for each cell.
    get_measured_flows() -> dict[int, float]:
        Returns a dictionary mapping measuring point IDs to their flow rates.
    _compute_distance_grid(targets: tuple[utils.Position]) -> npt.NDArray[np.float64]:
        Computes a grid of distances from each cell to the closest target using a specified algorithm.
    _compute_naive_distance_grid(targets: tuple[utils.Position]) -> npt.NDArray[np.float64]:
        Calculates distances to targets without considering obstacles.
    _compute_dijkstra_distance_grid(targets: tuple[utils.Position]) -> npt.NDArray[np.float64]:
        Calculates distances to targets using Dijkstra's algorithm, avoiding obstacles.
    _get_neighbors(position: utils.Position, shuffle: bool = True) -> list[utils.Position]:
        Returns a list of neighboring cells for a given position.
    _post_process():
        Saves the simulation output for analysis if an output filename is provided.
    """

    def __init__(self, config: el.SimulationConfig, random_seed: int = 42):
        """
        Initializes the Simulation class with the given configuration and random seed.

        Parameters:
        -----------
        config : el.SimulationConfig
            A configuration object that specifies the simulation parameters, including the
            grid size, initial positions of pedestrians, obstacles, targets, and the
            distance computation method.
        random_seed : int, optional
            The seed for the random number generator to ensure reproducibility of random operations.
            Default is 42.
        """

        self.grid_size = config.grid_size
        self.width, self.height = self.grid_size.width, self.grid_size.height
        self.grid = np.full(
            (self.width, self.height), el.ScenarioElement.empty
        )
        self.pedestrians = config.pedestrians
        self.obstacles = config.obstacles
        self.targets = config.targets
        self.distance_computation = config.distance_computation
        self.output_filename = config.output_filename
        self.is_absorbing = config.is_absorbing
        self.measuring_points = config.measuring_points
        self.get_grid()
        self.reached_pedestrians = list()
        self.current_step = 0
        self.mp_compute_obstacles_inside()

    def update(self, perturb: bool = True) -> bool:
        """Performs one step of the simulation, updating pedestrian positions on the grid.

        This method calculates the distance from each pedestrian to the target,
        determines potential moves based on neighboring cells, and updates the
        grid state accordingly. Pedestrians can either be updated in a fixed
        order or shuffled before each update, depending on the perturb parameter.

        Arguments:
        ----------
        perturb : bool
            If True, the pedestrians' positions are shuffled before updating,
            allowing for random movement. If False, the update occurs in a fixed
            order based on their current positions.

        Returns:
        -------
        bool
            Returns True if all pedestrians have reached their target position,
            indicating that the simulation is complete. Returns False if at least
            one pedestrian has not yet reached the target.
        """
        if not self.targets:
            print("No target is available. The pedestrians will not be moved.")
            return False

        targets_already_entered = []
        distance_grid = self._compute_distance_grid(self.targets)

        if perturb:
            random.shuffle(self.pedestrians)

        self.mp_reset_and_check_active()
        self.mp_compute_density()

        for pedestrian in self.pedestrians[:]:
            pedestrian.distances_made_this_step = 0

            if self.is_pedestrian_absorbed(
                pedestrian, targets_already_entered
            ):
                continue

            pedestrian.steps_already_made += 1

            pedestrian.speed_dec_acc += pedestrian.speed % 1
            for _ in range(
                int(pedestrian.speed) + int(pedestrian.speed_dec_acc)
            ):
                if pedestrian.speed_dec_acc >= 1:
                    pedestrian.speed_dec_acc -= 1

                best_neighbor = self.get_next_move_position(
                    pedestrian, distance_grid
                )
                x_dif, y_dif = (
                    best_neighbor.x - pedestrian.x,
                    best_neighbor.y - pedestrian.y,
                )

                # Diagonal movement logic
                if x_dif * y_dif != 0:
                    pedestrian.diagonal_encountered += 1
                    if (
                        pedestrian.diagonal_encountered != 0
                        and pedestrian.diagonal_encountered % 3 == 0
                    ):
                        continue

                # Update movement
                if (
                    self.is_absorbing
                    and self.grid[best_neighbor.x, best_neighbor.y]
                    == el.ScenarioElement.target
                ):
                    if (
                        best_neighbor.x,
                        best_neighbor.y,
                    ) not in targets_already_entered:
                        self.grid[pedestrian.x, pedestrian.y] = (
                            el.ScenarioElement.empty
                        )
                        self.compute_ped_distance(
                            pedestrian=pedestrian, x_dif=x_dif, y_dif=y_dif
                        )
                        targets_already_entered.append(
                            (best_neighbor.x, best_neighbor.y)
                        )
                        pedestrian.x, pedestrian.y = (
                            best_neighbor.x,
                            best_neighbor.y,
                        )
                    else:
                        continue
                else:
                    self.grid[pedestrian.x, pedestrian.y] = (
                        el.ScenarioElement.empty
                    )
                    self.grid[best_neighbor.x, best_neighbor.y] = (
                        el.ScenarioElement.pedestrian
                    )
                    self.compute_ped_distance(pedestrian, x_dif, y_dif)
                    pedestrian.x, pedestrian.y = (
                        best_neighbor.x,
                        best_neighbor.y,
                    )

                if pedestrian.has_reached_target(self.targets):
                    self.reached_pedestrians.append(pedestrian)
                    break

        self.mp_compute_speed()
        self.print_mp_result()

        self.current_step += 1

        return all(
            pedestrian.has_reached_target(self.targets)
            for pedestrian in self.pedestrians
        )

    def is_pedestrian_absorbed(self, pedestrian, targets_already_entered):
        """Check if a pedestrian is absorbed by a target and removed from the pedestrian list."""
        if pedestrian.has_reached_target(self.targets):
            pedestrian.took_steps = pedestrian.steps_already_made
            self.reached_pedestrians.append(pedestrian)
            if self.is_absorbing:
                self.pedestrians.remove(pedestrian)
                targets_already_entered.append((pedestrian.x, pedestrian.y))
            return True
        return False

    def mp_reset_and_check_active(self):
        """Reset attributes of measuring points and check if they are active for the current step."""
        for mp in self.measuring_points:
            mp.peds_within_area = 0
            mp.peds_to_track = []
            mp.peds_total_distances = 0
            if mp.delay <= self.current_step < mp.delay + mp.measuring_time:
                mp.measuring_active = True
            else:
                mp.measuring_active = False

    def mp_compute_obstacles_inside(self):
        for mp in self.measuring_points:
            for obstacle in self.obstacles:
                if mp.is_within_area(obstacle):
                    mp.obstacles_inside += 1

    def mp_compute_density(self):
        """Calculate pedestrian density in each measuring point area if it is active."""
        for ped in self.pedestrians:
            for mp in self.measuring_points:
                if mp.is_within_area(ped) and mp.measuring_active:
                    mp.peds_within_area += 1
                    mp.peds_to_track.append(ped.ID)
        for mp in self.measuring_points:
            if mp.measuring_active:
                mp.density_data.append(
                    mp.peds_within_area
                    / (mp.size.width * mp.size.height - mp.obstacles_inside)
                )

    def mp_compute_speed(self):
        """Compute the average speed of tracked pedestrians within active measuring points."""
        for mp in self.measuring_points:
            if mp.measuring_active:
                for ped in self.pedestrians:
                    if ped.ID in mp.peds_to_track:
                        mp.peds_total_distances += ped.distances_made_this_step
                if len(mp.peds_to_track) == 0:
                    mp.speed_data.append(0)
                else:
                    mp.speed_data.append(
                        mp.peds_total_distances / len(mp.peds_to_track)
                    )

    def compute_ped_distance(self, pedestrian, x_dif, y_dif):
        """Update pedestrian's distance based on movement in the x and y directions."""
        if x_dif * y_dif != 0:
            pedestrian.distances_made_this_step += np.sqrt(2)
        elif (x_dif != 0 and y_dif == 0) or (x_dif == 0 and y_dif != 0):
            pedestrian.distances_made_this_step += 1

    def print_mp_result(self):
        """Print the results for each measuring point at the end of its measuring period."""
        for mp in self.measuring_points:
            if self.current_step == mp.delay + mp.measuring_time - 1:
                mean_flow = mp.get_mean_flow()
                print(
                    "mp-ID:",
                    mp.ID,
                    "\n",
                    "mean speed:",
                    mp.mean_speed,
                    "\n",
                    "mean density:",
                    mp.mean_density,
                    "\n",
                    "mean flow:",
                    mean_flow,
                )

    def get_next_move_position(self, pedestrian, distance_grid):
        """Determine the next move position for a pedestrian based on the distance grid."""
        ped_x, ped_y = pedestrian.x, pedestrian.y
        ped = el.ScenarioElement.pedestrian

        if pedestrian.has_reached_target(self.targets) or np.isinf(
            distance_grid[ped_x, ped_y]
        ):
            return pedestrian.get_position()
        else:
            neighbors_positions = self._get_neighbors(
                pedestrian.get_position()
            )
            for pos in neighbors_positions[:]:
                if (
                    self.grid[pos.x, pos.y] == ped
                    or distance_grid[pos.x, pos.y]
                    >= distance_grid[ped_x, ped_y]
                ):
                    neighbors_positions.remove(pos)
            if len(neighbors_positions) == 0:
                return pedestrian.get_position()
            else:
                neighbors_distances = [
                    distance_grid[neighbor.x, neighbor.y]
                    for neighbor in neighbors_positions
                ]
                min_dist = min(neighbors_distances)
                min_dist_indices = [
                    i
                    for i, dist in enumerate(neighbors_distances)
                    if dist == min_dist
                ]
                if len(min_dist_indices) == 1:
                    return neighbors_positions[min_dist_indices[0]]
                else:
                    # Several neighbors have the same lowest distance to target
                    len_to_move = []  # length needed for moving to the neighbor
                    for i in min_dist_indices:
                        len_to_move.append(
                            np.abs(neighbors_positions[i].x - ped_x)
                            + np.abs(neighbors_positions[i].y - ped_y)
                        )
                    min_ind_in_len_to_move = np.argmin(np.array(len_to_move))
                    return neighbors_positions[
                        min_dist_indices[min_ind_in_len_to_move]
                    ]

    def get_grid(self) -> npt.NDArray[el.ScenarioElement]:
        """Returns a full state grid of the shape (width, height)."""

        for target in self.targets:
            x, y = target.x, target.y
            self.grid[x, y] = el.ScenarioElement.target

        for pedestrian in self.pedestrians:
            x, y = pedestrian.x, pedestrian.y
            self.grid[x, y] = el.ScenarioElement.pedestrian

        for obstacle in self.obstacles:
            x, y = obstacle.x, obstacle.y
            self.grid[x, y] = el.ScenarioElement.obstacle

        return self.grid

    def get_distance_grid(self) -> npt.NDArray[np.float64]:
        """Returns a grid with distances to a closest target."""
        distance_grid = self._compute_distance_grid(self.targets)
        return distance_grid

    def get_measured_flows(self) -> dict[int, float]:
        """Returns a map of measuring points' ids to their flows.

        Returns:
        --------
        dict[int, float]
            A dict in the form {measuring_point_id: flow}.
        """
        return {mp.ID: mp.get_mean_flow() for mp in self.measuring_points}

    def _compute_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes a distance grid from all cells to the closest target using a specified algorithm.

        This method calculates the distance from each cell in the grid to the nearest target.
        Depending on the configuration, it uses one of two distance computation algorithms:
        "naive" (Euclidean distance without obstacles) or "Dijkstra" (which accounts for
        obstacles). If no targets are provided, it returns a grid filled with zeros.

        Parameters:
        -----------
        targets : tuple[utils.Position]
            A tuple of positions (x, y) representing the locations of targets on the grid.

        Returns:
        --------
        npt.NDArray[np.float64]
            A 2D array (of shape grid width x grid height) representing the distance of
            each cell to the closest target. Cells with obstacles may have infinite distance
            if using the Dijkstra algorithm.

        Notes:
        ------
        - "Naive" distance computes the Euclidean distance between each grid cell and the target.
        - "Dijkstra" distance calculates the shortest path while considering obstacles and
        assigning infinite distance to cells blocked by obstacles.
        - If no targets are provided, the method returns a grid filled with zeros.
        """

        if len(targets) == 0:
            distances = np.zeros((self.width, self.height))
            return distances

        match self.distance_computation:
            case "naive":
                distances = self._compute_naive_distance_grid(targets)
            case "dijkstra":
                distances = self._compute_dijkstra_distance_grid(targets)
            case _:
                print(
                    "Unknown algorithm for computing the distance grid: "
                    f"{self.distance_computation}. Defaulting to the "
                    "'naive' option."
                )
                distances = self._compute_naive_distance_grid(targets)
        return distances

    def _compute_naive_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes a distance grid without considering obstacles.

        Arguments:
        ----------
        targets : Tuple[utils.Position]
            A tuple of targets on the grid. For each cell, the algorithm
            computes the distance to the closes target.

        Returns:
        --------
        npt.NDArray[np.float64]
            An array of distances of the same shape as the main grid.
        """

        targets = [[*target] for target in targets]
        targets = np.vstack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        distances = scipy.spatial.distance.cdist(targets, positions)

        distances = np.min(distances, axis=0)
        distances = distances.reshape((self.height, self.width)).T

        for obstacle in self.obstacles:
            distances[obstacle.x, obstacle.y] = np.inf

        return distances

    def _compute_dijkstra_distance_grid(
        self, targets: tuple[utils.Position]
    ) -> npt.NDArray[np.float64]:
        """Computes a distance grid using Dijkstra's algorithm, avoiding obstacles.

        Arguments:
        ----------
        targets : Tuple[utils.Position]
            A tuple of target positions on the grid. For each cell, the algorithm computes
            the shortest path distance to the closest target while avoiding obstacles.

        Returns:
        --------
        npt.NDArray[np.float64]
            An array of distances of the same shape as the main grid. Cells that are obstacles
            have an infinite distance.
        """

        distances = np.full((self.width, self.height), np.inf)
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, 1),
            (1, 1),
            (-1, -1),
            (1, -1),
        ]

        for target in targets:
            distances[target.x, target.y] = 0

            pq = [(0, target.x, target.y)]
            while pq:
                current_dist, x, y = heapq.heappop(pq)

                if current_dist > distances[x, y]:
                    continue

                for direction in directions:
                    nx, ny = x + direction[0], y + direction[1]
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and self.grid[nx, ny] != el.ScenarioElement.obstacle
                    ):
                        if direction[0] * direction[1] == 0:
                            new_dist = current_dist + 1

                        else:
                            new_dist = current_dist + 2**0.5

                        if new_dist < distances[nx, ny]:
                            distances[nx, ny] = new_dist
                            heapq.heappush(pq, (new_dist, nx, ny))
        return distances

    def _get_neighbors(
        self, position: utils.Position, shuffle: bool = True
    ) -> list[utils.Position]:
        """Returns a list of neighboring cells for the position.

        Arguments:
        ----------
        positions : utils.Position
            A position on the grid.
        shuffle : bool
            An indicator if neighbors should be shuffled or returned
            in the fixed order.

        Returns:
        --------
        list[utils.Position]
            An array of neighboring cells. Two cells are neighbors
            if they share a common vertex.
        """

        x, y = (
            position.x,
            position.y,
        )
        possible_moves = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
        ]

        neighbors = [
            utils.Position(nx, ny)
            for nx, ny in possible_moves
            if 0 <= nx < self.width and 0 <= ny < self.height
        ]

        if shuffle:
            np.random.shuffle(neighbors)

        return neighbors

    def _post_process(self):
        """Stores the simulation output for analysis if an output filename is provided."""

        if self.output_filename is None:
            return

        np.save(self.output_filename, self.grid)
        print(f"Simulation results saved to {self.output_filename}.")
