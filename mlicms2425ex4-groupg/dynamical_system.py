from typing import List, Callable, Union
from numpy.typing import ArrayLike
import numpy as np
import scipy
from scipy.integrate import solve_ivp


class DynamicalSystem:
    """This class defines a dynamical system

    Methods
    --------
    solve_system(fun: Callable, init_state: ArrayLike, t_eval: ArrayLike):
        Solve the dynamical system
    """

    def __init__(self, xrange: List[List[int]], discrete: bool = False):
        """Parameters
        -------------
        xrange: List[List[int]]
            Is used to set the domain of the dynamical system.
            Specify start, end, number of points for each dimension
        discrete: bool = False
            If true, the dynamical system is time-discrete
        """
        self.xrange = xrange
        self.discrete = discrete
        self.X = self._set_grid_coordinates(xrange=self.xrange)

    def solve_system(self, init_state: ArrayLike, t_eval: ArrayLike) -> ArrayLike:
        """Solve the dynamical system

        Given the evolution rules, the initial point, and the time steps, we
        obtain the trajectory of the point. The solving method is different
        for time-discrete system, so two methods are implemented here.

        Parameters
        ----------
        init_state: ArrayLike
            Initial state of the system
        t_eval: ArrayLike
            Time steps of the trajectory

        Returns
        -------
        trajectory: ArrayLike
            Trajectory of the inital point in time
        """
        if not self.discrete:
            trajectory = solve_ivp(
                fun=self.fun,
                t_span=(t_eval[0], t_eval[-1]),
                y0=init_state,
                t_eval=t_eval,
            )
            return trajectory.y
        else:
            current_state = init_state
            trajectory = []
            for i in range(len(t_eval)):
                trajectory.append(current_state)
                current_state = self.fun(t_eval[i], current_state)
            return trajectory

    def _set_grid_coordinates(self, xrange: List[List[int]]) -> List[np.ndarray]:
        """Set up the coordinates. For multidimensional cases use meshgrid"""
        match len(xrange):
            case 1:
                return np.linspace(xrange[0][0], xrange[0][1], xrange[0][2])
            case 2:
                X1, X2 = np.meshgrid(
                    np.linspace(xrange[0][0], xrange[0][1], xrange[0][2]),
                    np.linspace(xrange[1][0], xrange[1][1], xrange[1][2]),
                )
                return [X1, X2]
            case 3:
                X1, X2, X3 = np.meshgrid(
                    np.linspace(xrange[0][0], xrange[0][1], xrange[0][2]),
                    np.linspace(xrange[1][0], xrange[1][1], xrange[1][2]),
                    np.linspace(xrange[2][0], xrange[2][1], xrange[2][2]),
                )
                return [X1, X2, X3]


class Task1(DynamicalSystem):
    """A linear dynamical system of the form dx/dt = Ax.

    This class implements a linear dynamical system where the evolution
    of the state is governed by multiplication with a parameter matrix A.
    For a scalar case, it reduces to multiplication by a scalar parameter.

    Parameters
    ----------
    par : ArrayLike
        Parameter matrix A for the linear system dx/dt = Ax.
        Can also be a scalar for 1D systems.
    *args
        Variable length argument list for DynamicalSystem parent class.
    **kwargs
        Arbitrary keyword arguments for DynamicalSystem parent class.

    Attributes
    ----------
    par : ArrayLike
        Stored parameter matrix A or scalar parameter.
    """

    def __init__(self, par, *args, **kwargs):
        """Initialize the linear dynamical system.

        Parameters
        ----------
        par : ArrayLike
            Parameter matrix A for the linear system dx/dt = Ax.
            Can also be a scalar for 1D systems.
        *args
            Variable length argument list for DynamicalSystem parent class.
        **kwargs
            Arbitrary keyword arguments for DynamicalSystem parent class.
        """
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Compute the vector field of the linear system.

        This method implements the vector field for a linear dynamical system
        of the form dx/dt = Ax, where A is the parameter matrix and x is the
        current state. For scalar systems, it reduces to dx/dt = ax where a
        is a scalar parameter. When dealing with meshgrid coordinates, it
        properly reshapes arrays to maintain compatibility.

        Parameters
        ----------
        t : float
            Time parameter (not used in autonomous system)
        x : ArrayLike
            Current state vector or scalar state value

        Returns
        -------
        ArrayLike
            The vector field dx/dt = Ax at point x
            For scalar x, returns ax where a is the scalar parameter
        """
        if np.isscalar(x):
            return self.par * x

        x = np.array(x)
        if len(x.shape) > 1:  # Handle meshgrid case
            original_shape = x.shape
            if len(original_shape) == 3:  # 2D meshgrid case
                # Reshape to (2, N*N) for matrix multiplication
                n = original_shape[1] * original_shape[2]
                x_reshaped = x.reshape(2, n)
                # Compute the matrix multiplication
                result = np.dot(self.par, x_reshaped)
                # Reshape back to original shape
                return result.reshape(original_shape)
            return x  # Return unchanged if not in expected format

        return np.dot(self.par, x)  # Regular vector case


class Task21(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a float/int here"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Compute the vector field of the saddle-node system.

        Parameters
        ----------
        t : float
            Time parameter (not used in autonomous system)
        x : ArrayLike
            Current state

        Returns
        -------
        ArrayLike
            The vector field dx/dt = α - x² at point x
        """
        return self.par - x**2


class Task22(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a float/int here"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Compute the vector field of the shifted saddle-node system.

        Parameters
        ----------
        t : float
            Time parameter (not used in autonomous system)
        x : ArrayLike
            Current state

        Returns
        -------
        ArrayLike
            The vector field dx/dt = α - 2x² - 3 at point x
        """
        return self.par - 2 * x**2 - 3


class Task3(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a float/int here

        Initialize Task 3's system.

        Args:
            par (float or int): The parameter controlling the system.
            *args: Additional positional arguments for the base class.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(*args, **kwargs)
        self.par = par
        self.fun = self._hopf_bifurcation_system

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Wrapper method to call the Hopf bifurcation system."""
        return self._hopf_bifurcation_system(t, x)

    def _hopf_bifurcation_system(self, t: float, x: ArrayLike) -> ArrayLike:
        """Implement the system that exhibits Hopf bifurcation
        Args:
            t (float): Time
            x (ArrayLike): State vector (x1, x2)

        Returns:
            ArrayLike: The time derivative (dx1/dt, dx2/dt)
        """
        x1, x2 = x
        r_squared = x1**2 + x2**2
        dx1_dt = self.par * x1 - x2 - r_squared * x1
        dx2_dt = x1 + self.par * x2 - r_squared * x2
        return [dx1_dt, dx2_dt]


class Task41(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a float/int here, which represents r in equation x_n+1 = r*x_n*(1-x_n)"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """implements the discrete system x_n+1 = r*x_n*(1-x_n).

        Parameters
        ----------
        t : float
            Time parameter (not used in this system)
        x : ArrayLike
            Current state

        Returns
        -------
        ArrayLike
            The next state x_n+1 = r*x_n*(1-x_n)
        """
        x_next = self.par * x * (1 - x)
        return x_next


class Task42(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a list [sigma, beta, rho], which are parameters of Lorenz system"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Implements the Lorenz system
        Args:
            t (float): Time
            x (ArrayLike): State vector (x, y, z)

        Returns:
            ArrayLike: The results of Lorenz equations, i.e. time derivative (dx_dt, dy_dt, dz_dt)
        """
        sigma, beta, rho = self.par

        # the name x_lor is used for differentiating between function argument x and x of lorenz equation (i.e. x_lor)
        x_lor, y, z = x

        # apply 3 ordinary differential equations, also known as the Lorenz equations
        dx_dt = sigma * (y - x_lor)
        dy_dt = x_lor * (rho - z) - y
        dz_dt = (x_lor * y) - (beta * z)

        return [dx_dt, dy_dt, dz_dt]
