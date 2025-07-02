from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

###############################################################################
# Data loading
###############################################################################


def load_function(filename: str) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Loads evaluations of f(x) from a file.

    Parameters:
    -----------
    filename : str
        Path to the file containing the data. The data should have
        d + 1 columns, where the first d columns are the input data
        and the last column is the target data.

    Returns:
    --------
    tuple[npt.ArrayLike, npt.ArrayLike]
        A tuple of input data of shape (N, d) and target data of
        shape (N,), where N is the number of rows in the file.
    """
    data = np.loadtxt(filename)
    x, y = data[:, :-1], data[:, -1]
    return x, y


def load_vectorfield(filename: str) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Loads evaluations of a dynamical system from a file.

    Parameters:
    -----------
    filename : str
        Path to the file containing the data. The data should
        have 2*d columns, where the first d columns are the
        x0 coordinates and the last d columns are the x1
        coordinates.

    Returns:
    --------
    tuple[npt.ArrayLike, npt.ArrayLike], shape [(N, d), (N, d)]
        x0 and x1 positions for the dynamical system.
    """
    data = np.loadtxt(filename)
    if data.shape[1] % 2 != 0:
        raise ValueError("The number of columns in the file should be even.")
    d = data.shape[1] // 2
    x0, x1 = data[:, :d], data[:, d:]
    return x0, x1


def load_manifold(filename: str) -> npt.ArrayLike:
    """Loads coordinates of manifold points from a file.

    Parameters:
    -----------
    filename : str
        Path to the file containing the data.

    Returns:
    --------
    npt.ArrayLike
        Array of the shape (N, d), where N is the neumber of row
        in the file and d is the dimensionality of the embedding space.
    """
    data = np.loadtxt(filename)
    return data


###############################################################################
# Plotting. You can change these functions to adjust your plots.
###############################################################################


def plot_function(
    ax: plt.Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    y_pred: npt.ArrayLike | None = None,
    scatter: bool = False,
    xlabel: str = "x",
    ylabel: str = "y",
):
    """Plots a 1D function and its prediction.

    Parameters:
    -----------
        ax: plt.Axes
            A matplotlib axis to plot on.
        x: npt.ArrayLike
            A 1D array of arguments.
        y: npt.ArrayLike
            A 1D array of function values.
        y_pred: npt.ArrayLike | None = None
            An optional 1D array of predicted values.
        scatter: bool = False
            If true, use ax.scatter instead of ax.plot.
        xlabel: str = "x"
            The label of the x-axis.
        ylabel: str = "y"
            The label of the y-axis.
    """
    if scatter:
        plotter = partial(ax.scatter, s=1)
    else:
        plotter = ax.plot
        # Sort the arguments.
        argsort = np.argsort(x.flatten())
        x = x[argsort]
        y = y[argsort]
        if y_pred is not None:
            y_pred = y_pred[argsort]

    plotter(x, y, label="True labels")
    if y_pred is not None:
        plotter(x, y_pred, label="Predicted labels")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_vector_field(ax: plt.Axes, x0: npt.ArrayLike, x1: npt.ArrayLike, x_label="x[0]", y_label="x[1]"):
    """Plots a 2D vector field.

    Parameters:
    -----------
    ax: plt.Axes
        A matplotlib axis to plot on.
    x0: npt.ArrayLike
        Initial states of the dynamical system.
    x1: npt.ArrayLike
        Advanced states.
    """
    ax.scatter(x0[:, 0], x0[:, 1], s=1)
    ax.scatter(x1[:, 0], x1[:, 1], s=1)
    ax.quiver(
        x0[:, 0],
        x0[:, 1],
        x1[:, 0] - x0[:, 0],
        x1[:, 1] - x0[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_3D_trajectory(
    ax: plt.Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
):
    """Plots a 3D trajectory of a dynamical system

    Parameters:
    -----------
    ax: plt.Axes
        A matplotlib axis to plot on.
    x: npt.ArrayLike
        The x-coordinate of the trajectory
    y: npt.ArrayLike
        The y-coordinate of the trajectory
    z: npt.ArrayLike
        The z-coordinate of the trajectory
    c: npt.ArrayLike | None, default = None
        If present, this array is used to speficy colors of the points.
    """
    ax.scatter(x, y, z, s=1, c=c)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def plot_coordinate_vs_time(data, delay=0, coordinate_index=0):
    """
    Plots the coordinate against time and its delayed version.

    Parameters:
    -----------
    data (array): 
        The input data array where each column represents a different coordinate.
    delay (int): 
        The time delay to create a delayed version of the coordinate (default is 0).
        If delay = 0, plot the coordinate against time. If > 0, plot coordinate against
        its delayed version.
    coordinate_index (int): 
        The index of the coordinate to plot (default is 0, which corresponds to the first column).
    """
    coordinate = data[:, coordinate_index]
    
    # Plot coordinate vs time (row number)
    if delay == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(coordinate)
        plt.title(f"Coordinate {coordinate_index + 1} vs Time (Row Number)")
        plt.xlabel("Time (Row number)")
        plt.ylabel(f"Coordinate {coordinate_index + 1}")
        plt.grid(True)
        plt.show()

    # Plot coordinate vs delayed version
    if delay > 0:
        delayed_coordinate = coordinate[delay:]
        original_coordinate = coordinate[:-delay]
        
        plt.figure(figsize=(10, 6))
        plt.plot(original_coordinate, delayed_coordinate)
        plt.title(f"Coordinate {coordinate_index + 1} vs Delayed Coordinate (∆n={delay})")
        plt.xlabel(f"Coordinate {coordinate_index + 1} at time t")
        plt.ylabel(f"Delayed Coordinate at time t-{delay}")
        plt.grid(True)
        plt.show()


def plot_lorenz_attractors(x, y, z, delta_t):
    """
    Plots the original Lorenz attractor and its reconstructed versions from the x and z coordinates.
    
    Parameters:
    x (array): Array of x-values from the Lorenz system solution.
    y (array): Array of y-values from the Lorenz system solution.
    z (array): Array of z-values from the Lorenz system solution.
    delta_t (int): Time delay used for reconstructing the attractors.
    
    The function generates three subplots:
    1. The original Lorenz attractor (x, y, z).
    2. The reconstructed attractor from the x-coordinate (x, x(t+Δt), x(t+2Δt)).
    3. The reconstructed attractor from the z-coordinate (z, z(t+Δt), z(t+2Δt)).
    """
    # Reconstruct the attractor for the x-coordinate
    x1 = x[:-2*delta_t]
    x2 = x[delta_t:-delta_t]
    x3 = x[2*delta_t:]

    # Reconstruct the attractor for the z-coordinate
    z1 = z[:-2*delta_t]
    z2 = z[delta_t:-delta_t]
    z3 = z[2*delta_t:]

    fig = plt.figure(figsize=(14, 8))

    # Plot the original Lorenz attractor (x, y, z)
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(x, y, z, color='g', label="Original Lorenz Attractor")
    ax1.set_title("Original Lorenz Attractor (x, y, z)")
    ax1.set_xlabel("x(t)")
    ax1.set_ylabel("y(t)")
    ax1.set_zlabel("z(t)")
    ax1.legend()

    # Plot the reconstructed attractor from the x-coordinate
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot(x1, x2, x3, color='r', label="Reconstructed (x, x(t+Δt), x(t+2Δt))")
    ax2.set_title("Reconstructed Lorenz Attractor (x-coordinate)")
    ax2.set_xlabel("x(t)")
    ax2.set_ylabel("x(t+Δt)")
    ax2.set_zlabel("x(t+2Δt)")
    ax2.legend()

    # Plot the reconstructed attractor from the z-coordinate
    ax3 = fig.add_subplot(234, projection='3d')
    ax3.plot(z1, z2, z3, color='b', label="Reconstructed (z, z(t+Δt), z(t+2Δt))")
    ax3.set_title("Reconstructed Lorenz Attractor (z-coordinate)")
    ax3.set_xlabel("z(t)")
    ax3.set_ylabel("z(t+Δt)")
    ax3.set_zlabel("z(t+2Δt)")
    ax3.legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    plt.show()

def plot_3d_points(data, x_label = 'x1', y_label= 'x2', z_label= 'x3'):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=z, cmap='viridis', marker='.')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    plt.show()

def plot_2d_points(data, x_label = 'x1', y_label= 'x2', title = '2D Points Plot'):
    x = data[:, 0]
    y = data[:, 1]

    plt.plot(x, y, color='blue', linestyle='-')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
###############################################################################
