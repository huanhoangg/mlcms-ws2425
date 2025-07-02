from typing import List, Tuple, Type
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from dynamical_system import *
from scipy.linalg import eigvals
from mpl_toolkits.mplot3d import Axes3D

# TODO: add all the util functions here, e.g. plot functions.


# ---------------------------------TASK 1---------------------------------
def create_matrix(a, b, c, d):
    """Create a 2x2 matrix for the linear system.

    Parameters:
        a (float): Top-left element of matrix
        b (float): Top-right element of matrix
        c (float): Bottom-left element of matrix
        d (float): Bottom-right element of matrix

    Returns:
        numpy.ndarray: A 2x2 matrix of the form [[a, b], [c, d]]

    Example:
        >>> create_matrix(1, 0, 0, -1)
        array([[ 1,  0],
               [ 0, -1]])
    """
    return np.array([[a, b], [c, d]])


def compute_eigenvalues(matrix):
    """Compute eigenvalues of a 2x2 matrix.

    Parameters:
        matrix (numpy.ndarray): A 2x2 matrix

    Returns:
        numpy.ndarray: Array containing the two eigenvalues

    Example:
        >>> compute_eigenvalues(np.array([[1, 0], [0, -1]]))
        array([ 1., -1.])
    """
    return eigvals(matrix)


def create_vector_field(matrix):
    """Create vector field data for phase portrait visualization.

    Generates a grid of points and computes the vector field values
    for the linear system dx/dt = Ax where A is the input matrix.

    Parameters:
        matrix (numpy.ndarray): A 2x2 matrix representing the linear system

    Returns:
        tuple: Four numpy arrays (X, Y, U, V) where:
            - X, Y are meshgrid coordinates
            - U, V are the vector field components at each point

    Notes:
        Uses a 20x20 grid over the region [-2, 2] × [-2, 2]
    """
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    U = matrix[0, 0] * X + matrix[0, 1] * Y
    V = matrix[1, 0] * X + matrix[1, 1] * Y

    return X, Y, U, V


# ---------------------------------TASK 2---------------------------------
def find_saddle_node_equilibria(alpha):
    """Find equilibrium points for the basic saddle-node system x' = α - x².

    Parameters:
        alpha (float): Bifurcation parameter

    Returns:
        list: Equilibrium points (empty list if α < 0, else [√α, -√α])

    Notes:
        Equilibria exist only when α ≥ 0, corresponding to the points
        where α = x².
    """
    if alpha >= 0:
        return [np.sqrt(alpha), -np.sqrt(alpha)]
    return []


def find_shifted_saddle_node_equilibria(alpha):
    """Find equilibrium points for the shifted saddle-node system x' = α - 2x² - 3.

    Parameters:
        alpha (float): Bifurcation parameter

    Returns:
        list: Sorted list of real equilibrium points

    Notes:
        Solves the quadratic equation 2x² + (α - 3) = 0
        Returns only real roots, sorted in ascending order
    """
    coeffs = [-2, 0, (alpha - 3)]
    roots = np.roots(coeffs)
    return sorted([float(r) for r in roots if np.isreal(r)])


def evaluate_basic_stability(x, alpha):
    """Evaluate stability of equilibrium points in basic saddle-node system.

    Parameters:
        x (float): Equilibrium point to evaluate
        alpha (float): Bifurcation parameter

    Returns:
        float: Derivative df/dx = -2x at the equilibrium point
        Negative values indicate stability, positive values instability
    """
    return -2 * x


def evaluate_shifted_stability(x, alpha):
    """Evaluate stability of equilibrium points in shifted saddle-node system.

    Parameters:
        x (float): Equilibrium point to evaluate
        alpha (float): Bifurcation parameter

    Returns:
        float: Derivative df/dx = -4x at the equilibrium point
        Negative values indicate stability, positive values instability
    """
    return -4 * x

# ---------------------------------TASK 3---------------------------------
def plot_phase_portrait(
    ode_system: list,
    alpha: float,
    plot_orbit=False,
    start_pos=(2, 0),
    time_step=1e-3,
    num_iterations=10000,
):
    """
    Plots the phase portrait of a given 2D ODE system and optionally overlays an orbit computed via Euler's method.

    Args:
        ode_system (list): A list of two strings representing the ODEs for dx1/dt and dx2/dt.
        alpha (float): Parameter in the system that governs its behavior.
        plot_orbit (bool): Whether to plot the orbit on top of the phase portrait (default is False).
        start_pos (tuple): Initial coordinates of the point (x1, x2) (default is (2, 0)).
        time_step (float): Step size (time step) for Euler's method (default is 1e-3).
        num_iterations (int): Number of iterations for the orbit (default is 10000).
    """

    if len(ode_system) != 2:
        raise ValueError("The system must contain exactly two ODEs.")

    grid_limit = 2.5
    grid_resolution = 50
    y_values, x_values = np.mgrid[
        -grid_limit : grid_limit : grid_resolution * 1j,
        -grid_limit : grid_limit : grid_resolution * 1j,
    ]

    x1_component, x2_component = [], []
    # DO NOT REMOVE x2 AND x1 BECAUSE EVEN THOUGH THEY ARE NOT USED IN THE LOOP, THEY
    # ARE NEEDED FOR THE OPERATION EVAL
    for x2 in x_values[0]:
        for x1 in y_values[:, 0]:
            dx1_dt = eval(ode_system[0])
            dx2_dt = eval(ode_system[1])
            x1_component.append(dx1_dt)
            x2_component.append(dx2_dt)

    x1_component = np.reshape(x1_component, x_values.shape)
    x2_component = np.reshape(x2_component, x_values.shape)

    plt.figure(figsize=(10, 10))
    plt.streamplot(x_values, y_values, x1_component, x2_component, density=2)

    if plot_orbit:
        orbit_x, orbit_y = [start_pos[0]], [start_pos[1]]
        x = np.array(start_pos)

        for _ in range(num_iterations):
            x1, x2 = x[0], x[1]

            u = eval(ode_system[0])
            v = eval(ode_system[1])

            # EULER's UPDATE STEP
            x = x + time_step * np.array([u, v])

            orbit_x.append(x[0])
            orbit_y.append(x[1])

        plt.plot(orbit_x, orbit_y, color="blue")
        plt.scatter(orbit_x[0], orbit_y[0], color="red")
        plt.legend(fontsize=12)

    plt.title(
        f"Starting point {start_pos}" if plot_orbit else f"Phase Portrait: α = {alpha}",
        fontsize=24,
    )
    plt.xlabel("x₁", fontsize=18)
    plt.ylabel("x₂", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    plt.show()
    plt.close()

def plot_cusp_bifurcation(x_range, alpha2_range, num_samples=2000):
    """
    Visualize the bifurcation surface (x, alpha1, alpha2) where x_dot = 0
    for the cusp bifurcation in a 3D plot from three different angles.
    
    Parameters:
    x_range : tuple
        The range of values for x
    alpha2_range : tuple
        The range of values for alpha2
    num_samples : int, optional
        The number of samples to take in the x and alpha2 ranges. Default is 200.
    """
    
    # Sample x and alpha2 uniformly from the specified ranges and number of samples
    x_vals = np.linspace(x_range[0], x_range[1], num_samples)
    alpha2_vals = np.linspace(alpha2_range[0], alpha2_range[1], num_samples)
    x_vals, alpha2_vals = np.meshgrid(x_vals, alpha2_vals)

    alpha1_vals = -alpha2_vals * x_vals + x_vals ** 3

    fig = plt.figure(figsize=(18, 8))

    # Plot 1 - View 1:
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(alpha1_vals, alpha2_vals, x_vals, c=x_vals, cmap='viridis', s=5)
    cbar1 = fig.colorbar(scatter1, ax=ax1)
    cbar1.set_label('x values')
    ax1.set_xlabel(r'$\alpha_1$')
    ax1.set_ylabel(r'$\alpha_2$')
    ax1.set_zlabel(r'$x$')
    ax1.set_title('Bifurcation Surface (View 1)')
    ax1.view_init(elev=30, azim=60)

    # Plot 2 - View 2:
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(alpha1_vals, alpha2_vals, x_vals, c=x_vals, cmap='viridis', s=5)
    cbar2 = fig.colorbar(scatter2, ax=ax2)
    cbar2.set_label('x values')
    ax2.set_xlabel(r'$\alpha_1$')
    ax2.set_ylabel(r'$\alpha_2$')
    ax2.set_zlabel(r'$x$')
    ax2.set_title('Bifurcation Surface (View 2)')
    ax2.view_init(elev=45, azim=90) 

    # Plot 3 - View 3:
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(alpha1_vals, alpha2_vals, x_vals, c=x_vals, cmap='viridis', s=5)
    cbar3 = fig.colorbar(scatter3, ax=ax3)
    cbar3.set_label('x values')
    ax3.set_xlabel(r'$\alpha_1$')
    ax3.set_ylabel(r'$\alpha_2$')
    ax3.set_zlabel(r'$x$')
    ax3.set_title('Bifurcation Surface (View 3)')
    ax3.view_init(elev=60, azim=120)

    plt.tight_layout()
    plt.show()
    plt.close() 


# ---------------------------------TASK 4---------------------------------

def plot_difference_lorenz_system(lorenz_system_first: DynamicalSystem, lorenz_system_second: DynamicalSystem, init_state_first: ArrayLike, init_state_second: ArrayLike,t_eval: ArrayLike, is_small_rho):

    """
    Plotting function that shows the differences between trajectories of Lorenz systems

    Parameters:
        lorenz_system_first (DynamicalSystem): First Lorenz system for computing trajectory differences
        lorenz_system_second (DynamicalSystem): First Lorenz system for computing trajectory differences
        init_state_first (ArrayLike): initial state of the first Lorenz system
        init_state_second (ArrayLike): initial state of the first Lorenz system
        is_small_rho (ArrayLike): depending on rho value (i.e. 28 (high rho) or 0.5 (small rho)) we apply different extra operations
        t_eval (ArrayLike): Time steps of the trajectory

    Returns:
        None: shows the plot of differences
    """

    trajectory_first = lorenz_system_first.solve_system(init_state_first, t_eval)
    trajectory_second = lorenz_system_second.solve_system(init_state_second, t_eval)

    difference_squared = np.sum((trajectory_first - trajectory_second) ** 2, axis=0)


    plt.figure(figsize=(12, 8))
    plt.plot(t_eval, difference_squared, color='blue', lw=0.5)

    if is_small_rho:
        # as recommended in https://stackoverflow.com/questions/53569576/how-to-draw-bar-charts-for-very-small-values-in-python-or-matplotlib
        plt.yscale('log')
    else:
        plt.axvline(x=25, color='red', linestyle='--')

    t_end = t_eval[-1]
    plt.xticks(np.arange(0, t_end, 50))
    plt.title("Lorenz System Trajectory Difference over Time")
    plt.xlabel("Time")
    plt.ylabel("Difference")
    plt.show()
    plt.close()

def plot_trajectory_lorenz_system(lorenz_system: DynamicalSystem, init_state: ArrayLike, t_eval: ArrayLike, line_width):

    """
    Plotting function that shows a single trajectory of the Lorenz system.

    Parameters:
        lorenz_system (DynamicalSystem): The Lorenz system whose trajectory is plotted
        init_state (ArrayLike): initial state of the Lorenz system
        t_eval (ArrayLike): Time steps of the trajectory
        line_width: Width of the line representing trajectory,
        used for making the line thinner or thicker depending on the plot for making it visually more clear

    Returns:
        None: shows the plot of Lorenz system trajectory
    """

    x, y, z = lorenz_system.solve_system(init_state, t_eval)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=line_width)
    ax.set_title(f"Lorenz System Trajectory initial condition: {init_state}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    plt.close()


# this function is not used in the report, it was just used for understanding the logistic map system better, i.e. x_n+1 = r*x_n*(1-x_n)
def plot_single_trajectory_t4(dynamical_system: DynamicalSystem, init_state: ArrayLike, t_eval: ArrayLike, title, save_loc):
    """
    Plotting function that shows a single trajectory of the dynamical system (logistic map in our case).

    Parameters:
        dynamical_system (DynamicalSystem): The dynamical system (logistic map here in our case) whose trajectory is plotted
        init_state (ArrayLike): initial state of the dynamical system
        t_eval (ArrayLike): Time steps of the trajectory
        title: title of the plot
        save_loc: location for saving the plot

    Returns:
        None: shows the plot of trajectory
    """

    states = dynamical_system.solve_system(init_state, t_eval)

    plt.figure(figsize=(16, 9))
    plt.plot(t_eval, states, "b.", markersize=5)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(title)
    plt.savefig(save_loc)
    plt.show()
    plt.close()


# You should use the same plot function for the bifurcation plots in task 2 and 4
def plot_bifurcation_diagram(
        alphas,
        system=None,
        find_equilibria=None,
        evaluate_stability=None,
        init_state=None,
        t_eval=None,
        num_transient_steps=None,
        critical_points=None,
        title="Bifurcation Diagram",
        xlim=None,
        ylim=None,
        save_loc=None,
        task_type="analytical"  # Can be either "analytical" (task 2) or "numerical" (task 4)
):
    """Unified bifurcation diagram plotting function that handles both analytical (Task 2)
    and numerical (Task 4) approaches.

    Parameters:
        alphas (array-like): Range of bifurcation parameter values to plot
        system (DynamicalSystem, optional): Required for numerical approach, dynamical system for computing trajectories
        find_equilibria (callable, optional): Required for analytical approach
        evaluate_stability (callable, optional): Required for analytical approach
        init_state (array-like, optional): Required for numerical approach, initial state of dynamical system
        t_eval (array-like, optional): Time steps of the trajectory
        num_transient_steps (int, optional): Required for numerical approach, number of initial transient steps to skip
        critical_points (list, optional): Points to mark with vertical red lines
        title (str): Title for the plot
        xlim (tuple, optional): (min, max) for x-axis limits
        ylim (tuple, optional): (min, max) for y-axis limits
        save_loc (str, optional): Location to save the plot
        task_type (str): Either "analytical" or "numerical"

    Returns:
        tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes) for analytical approach
        None: for numerical approach (saves and shows the plot directly)
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    if task_type == "analytical":
        # Implementation for Task 2
        stable_alphas, stable_x = [], []
        unstable_alphas, unstable_x = [], []

        for alpha in alphas:
            states = find_equilibria(alpha)
            for state in states:
                stab = evaluate_stability(state, alpha)
                if stab < 0:
                    stable_alphas.append(alpha)
                    stable_x.append(state)
                else:
                    unstable_alphas.append(alpha)
                    unstable_x.append(state)

        ax.plot(
            stable_alphas,
            stable_x,
            ".",
            color="blue",
            label="Stable",
            markersize=2,
            linestyle="none",
        )
        ax.plot(
            unstable_alphas,
            unstable_x,
            ".",
            color="red",
            label="Unstable",
            markersize=2,
            linestyle="none",
        )
        ax.legend()

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(True)

    else:  # task_type == "numerical"
        # Implementation for Task 4
        all_states = []
        corresponding_pars = []

        # here par represents the r in logistic map x_n+1 = r*x_n*(1-x_n), we plot the states with each r (par)
        # in the defined range. We skip the num_transient_steps amount of steps to skip the initial transient behaviour
        # states are accumulated in the array all_states and their corresponding r parameters (par),
        # with which they are computed, are stored in corresponding_pars
        for par in alphas:
            system.par = par
            current_states = system.solve_system(init_state, t_eval)
            current_states = current_states[num_transient_steps:]
            all_states.extend(current_states)
            corresponding_pars.extend([par] * len(current_states))

        ax.plot(corresponding_pars, all_states, "b.", markersize=0.02)

        # these critical_points represent the bifurcation points and here red vertical lines are drawn on them
        # because different steady states or limit cycles happen between them
        if critical_points:
            for point in critical_points:
                ax.axvline(x=point, color='red', linestyle='--')

    ax.set_xlabel(r"$\alpha$ (or r)")
    ax.set_ylabel(r"$x$")
    ax.set_title(title)

    plt.show()

    if task_type == "analytical":
        return fig, ax
    else:
        plt.close()
        return None