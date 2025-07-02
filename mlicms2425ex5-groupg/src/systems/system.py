import abc
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.integrate as spi
from scipy.linalg import lstsq
from sklearn.base import RegressorMixin
from scipy.integrate import solve_ivp

from ..models.approximator import BaseApproximator
from ..models.utils import *
from . import utils

class BaseDynamicalSystem(abc.ABC):
    """
    Abstract base class for representing dynamical systems.

    This class defines the interface for simulating dynamical systems and requires
    subclasses to implement the `_get_tangent` method. The `_get_tangent` method
    computes the tangent vector, which defines the evolution of the system at a given
    state and time. Additionally, this class provides methods for simulating the system
    for a single initial state or in batch mode for multiple initial conditions.

    Subclasses should override `_get_tangent` to specify the dynamics of the system.

    Methods:
    --------
        simulate(state, t_max, n_evals):
            Simulates the dynamical system for a given initial state and time interval.

        batch_simulate(states, t_max, n_evals):
            Simulates the dynamical system for multiple initial states over a time interval.

    Abstract Methods:
    -----------------
        _get_tangent(t, state):
            Computes the tangent vector corresponding to the dynamical system at a given state and time.
    """

    @abc.abstractmethod
    def _get_tangent(self, t: float, state: npt.ArrayLike) -> npt.ArrayLike:
        """Computes the tangent corresponding to the dynamical system.

        Parameters:
        -----------
            t: float
                Time at which the tangent is evaluated.
            state: npt.ArrayLike, shape (d,)
                State of the dynamical system.

        Returns:
        --------
            npt.ArrayLike, shape (d,)
                Tangent vector corresponding to the state.
        """
        pass

    def simulate(
        self, state: npt.ArrayLike, t_max: float, n_evals: int | None = None
    ) -> npt.ArrayLike:
        """Simulate the dynamical system for a given time.

        Parameters:
        -----------
            state: npt.ArrayLike, shape (d,)
                Initial state of the system.
            t_max: float
                Time to simulate the system.
            n_evals: int | None, default = None
                Number of timesteps to evaluate at. The timestamps are
                equally spaced between 0 and t. If n_evals is None, then
                only the final state is returned.

        Returns:
        --------
            npt.ArrayLike, shape (n_evals, d)
                States at evaluated timestamps.
        """
        if n_evals is not None:
            t_eval = np.linspace(0, t_max, n_evals)
        elif n_evals is None:
            t_eval = None

        sol = solve_ivp(self._get_tangent, [0, t_max], y0=state, t_eval=t_eval)

        if n_evals is not None:
            return np.transpose(sol.y)
        elif n_evals is None:
            return np.transpose(sol.y)[-1].reshape(1,-1)  # changing here causes test failure out of whatever reason


    def batch_simulate(
        self, states: npt.ArrayLike, t_max: float, n_evals: int = None
    ) -> npt.ArrayLike:
        """Simulate the dynamical system for a given time for several initial conditions.

        Parameters:
        -----------
            states: npt.ArrayLike, shape (N, d)
                Initial state of the system.
            t_max: float
                Time to simulate the system.
            n_evals: int
                Number of timesteps to evaluatte at. The timestamps are
                equally spaced between 0 and t.

        Returns:
        --------
            npt.ArrayLike, shape (N, n_evals, d) or (N, d)
                States ate evaluated timestamps. If n_evals is None,
                then only the final states are returned as an array
                of shape (N, d).
        """
        result_list = [self.simulate(each_state, t_max, n_evals) for each_state in states]
        result = np.array(result_list)

        # convert shape (N,1,D) to (N,D) in case n_eval is None
        if n_evals is None:
            result = result.reshape(result.shape[0], result.shape[2])
        return result


@dataclass
class LorenzSystem(BaseDynamicalSystem):
    """
    A class representing the Lorenz dynamical system.

    The Lorenz system is a set of three coupled, first-order, nonlinear differential equations.
    It is a classic example of a chaotic system and is commonly used in studies of deterministic chaos.

    Attributes:
    -----------
        sigma : float, default = 10
            The Prandtl number, controlling the rate of convection.
        rho : float, default = 28
            The Rayleigh number, controlling the difference in temperature.
        beta : float, default = 8/3
            A geometric factor.

    Methods:
    --------
        _get_tangent(_, state: npt.ArrayLike) -> npt.ArrayLike:
            Computes the tangent vector for the Lorenz system.
    """

    sigma: float = 10
    rho: float = 28
    beta: float = 8 / 3

    def _get_tangent(self, _, state: npt.ArrayLike) -> npt.ArrayLike:
        x, y, z = state
        result = np.array([self.sigma*(y-x), x*(self.rho-z)-y, x*y-self.beta*z])
        return result


@dataclass
class TrainableDynamicalSystem(BaseDynamicalSystem):
    """
    A trainable dynamical system that approximates its tangent map based on snapshots.

    This class allows for the inference and learning of a dynamical system's behavior
    by fitting a model to snapshot data (pairs of states at different time steps).
    The tangent map can then be used for simulation or analysis.

    Attributes:
    -----------
        approximator : BaseApproximator
            An approximator model for the dynamical system.
        weights : np.ndarray, default = None
            Learned weights for the tangent map, computed during training.
        tangent : np.ndarray, default = None
            Tangent vectors approximated during training.
        rcond : float, default = 1e-3
            Regularization parameter for least squares fitting.
    """

    approximator: BaseApproximator
    weights = None
    tangent = None
    rcond: float = 1e-3

    def _infer_tangent(
        self, x0: npt.ArrayLike, x1: npt.ArrayLike, delta_t: float
    ) -> npt.ArrayLike:
        """Approximates the tangent map of a dynamical system from snapshots.

        Parameters:
        -----------
            x0, x1: npt.ArrayLike, shape (N, d)
                N snapshots of the dynamical system.
            delta_t: float
                Time step between the snapshots.

        Returns:
        --------
            npt.ArrayLike, shape (N, d)
                N approximated tangent vectors.
                The i-th row of the output array is the approximation of the
                ith position.
        """
        # sanity checks
        assert x0.ndim == 2, f"x0 have shape {x0.shape}, while it should be (N, d)"
        assert x1.ndim == 2, f"x1 have shape {x0.shape}, while it should be (N, d)"
        assert  x0.shape == x1.shape, f"x0.shape != x1.shape, their shapes are {x0.shape} and {x1.shape}"

        self.tangent = (x1-x0)/delta_t
        return self.tangent

    def fit(
        self, x0: npt.ArrayLike, x1: npt.ArrayLike, delta_t: float
    ) -> "TrainableDynamicalSystem":
        """
        Fits the dynamical system's tangent map to the provided snapshot data.

        This method computes the tangent vectors and learns the system's weights
        using a least squares approach.

        Parameters:
        -----------
            x0, x1 : npt.ArrayLike, shape (N, d)
                N snapshots of the dynamical system at two consecutive time steps.
            delta_t : float
                Time step between the snapshots.

        Returns:
        --------
            TrainableDynamicalSystem
                The fitted dynamical system instance.
        """
        # sanity checks
        assert x0.ndim == 2, f"fit, x0 have shape {x0.shape}, while it should be (N, d)"
        assert x1.ndim == 2, f"fit, x1 have shape {x0.shape}, while it should be (N, d)"
        assert x0.shape == x1.shape, f"fit, x0.shape != x1.shape, their shapes are {x0.shape} and {x1.shape}"

        self._infer_tangent(x0, x1, delta_t)
        self.weights = linear_solve(x0, self.tangent, rcond=self.rcond)
        return self

    def _get_tangent(self, _, state: npt.ArrayLike) -> npt.ArrayLike:
        state_shape = state.shape

        result = None
        if state.ndim == 1:
            result = (state.reshape(1,-1) @ self.weights).reshape(state_shape)
        elif state.ndim == 2:
            result = state @ self.weights
            assert result.shape == state_shape, f"if state.ndim == 2, result shape is {result.shape} and state_shape is {state_shape}"
        return result


class GridSearchDynamicalSystem(RegressorMixin):
    """Wrapper for TrainableDynamicalSystem.

    This class wraps TrainableDynamicalSystem to make it compatible with
    the GridSearchCV.
    """

    def __init__(self, delta_t, approximator_cls, **approximator_params):
        self.delta_t = delta_t
        approximator = approximator_cls(**approximator_params)
        self.system = TrainableDynamicalSystem(approximator)

    def fit(self, x0, x1):
        self.system.fit(x0, x1, self.delta_t)
        return self

    def predict(self, x):
        return self.system.batch_simulate(x, self.delta_t, n_evals=2)[:, -1, :]

    def get_params(self, deep=True):
        approximator_cls = self.system.approximator.__class__
        approximator_params = self.system.approximator.get_params(deep)
        return approximator_params | {
            "delta_t": self.delta_t,
            "approximator_cls": approximator_cls,
        }

    def set_params(self, **params):
        self.__init__(**params)
        return self

