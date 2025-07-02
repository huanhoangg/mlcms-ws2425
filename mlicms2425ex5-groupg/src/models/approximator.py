import abc
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.base import RegressorMixin

from . import utils


@dataclass
class BaseApproximator(abc.ABC, RegressorMixin):
    """
    Base class for least-squares-based approximators.

    This class provides a framework for implementing approximators that use least squares
    to estimate model weights from input data and corresponding target values. Subclasses
    should implement the `construct_basis` method to define how the basis functions are
    constructed from input data.

    Attributes:
    -----------
        rcond : float, default = 0.0
            Cutoff for small singular values in the least squares solution. Used to
            regularize the computation of weights.
        _weights : npt.ArrayLike, default = None
            Model weights computed during the fitting process.
        x_basis : npt.ArrayLike, default = None
            Basis matrix constructed from input data during fitting.

    Methods:
    --------
        fit(x, y):
            Fits the approximator to the input data and target values using a least-squares solution.
        predict(x):
            Predicts target values for given input data using the learned weights.
        get_params():
            Retrieves model parameters.
        set_params(**params):
            Sets model parameters.
        construct_basis(x):
            Abstract method to define the basis construction for input data.
    """

    rcond: float = 0.0
    _weights: npt.ArrayLike = None
    x_basis = None

    @abc.abstractmethod
    def construct_basis(self, x: npt.ArrayLike) -> npt.ArrayLike:
        pass

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> "BaseApproximator":
        x, y = np.asarray(x), np.asarray(y)
        self.x_basis = self.construct_basis(x)
        self._weights = utils.linear_solve(self.x_basis, y, rcond=self.rcond)
        return self

    def predict(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Predict the target values for the given input data.
        Args:
            x : ndarray, shape (N, d_in)
            Feature matrix.
        Returns:
            y : ndarray, shape (N, d_out)
            Preditced target values.
        Raises:
            ValueError: If the model weights are not set.
        """
        if self._weights is None:
            raise ValueError(
                "The model weights are not set. "
                "You should call the fit(...) method first."
            )
        return self.construct_basis(x) @ self._weights

    def get_params(self, deep=True):
        return {"rcond": self.rcond}

    def set_params(self, **params):
        self.rcond = params["rcond"]
        return self


class LinearApproximator(BaseApproximator):
    """
    A basic linear least-squares approximator.

    This implementation uses the raw features of the input data as the basis for
    least-squares estimation. It assumes a simple linear relationship between the
    input features and the target values.
    """

    def construct_basis(self, x):
        """
        Constructs the basis matrix for linear approximation.
        In this basic implementation, the basis is identical to the input feature matrix.

        Parameters:
        -----------
            x : npt.ArrayLike, shape (N, d_in)
                Input feature matrix.

        Returns:
        --------
            npt.ArrayLike, shape (N, d_in)
                Basis matrix, which is the same as the input feature matrix.
        """
        return x


@dataclass
class RBFApproximator(BaseApproximator):
    """
    Radial Basis Function (RBF) Approximator.

    This approximator models the relationship between input features and target values
    using a radial basis function expansion. The basis functions are centered at
    selected data points, and their influence is scaled by a parameter `eps`.

    Attributes:
    -----------
        L : int, default = 10
            Number of RBF centers to use for constructing the basis.
        eps : float, default = 1e-1
            Scaling parameter that controls the width of the RBFs. Smaller values
            create narrower RBFs, leading to more localized influence.
        _centers : npt.ArrayLike, default = None
            Precomputed RBF centers selected from the input data.

    Methods:
    --------
        get_centers(x: npt.ArrayLike, seed: int = 42) -> npt.ArrayLike
            Selects RBF centers from the input data based on a random seed for reproducibility.

        construct_basis(x: npt.ArrayLike) -> npt.ArrayLike
            Constructs the RBF basis matrix for the input data using the precomputed centers.

        get_params(deep: bool = True) -> dict
            Retrieves the parameters of the approximator, including `L` and `eps`.

        set_params(**params) -> "RBFApproximator"
            Sets the parameters of the approximator, such as `L` and `eps`.
    """

    L: int = 10     # number of centers
    eps: float = 1e-1
    _centers: npt.ArrayLike = None

    def get_centers(self, x: npt.ArrayLike, seed: int = 42):
        """
        Select RBF centers from the input data.
        Args:
            x (npt.ArrayLike): Input data of shape (n_samples, n_features).
            seed (int): Random seed for reproducibility.
        Returns:
            npt.ArrayLike: Selected centers of shape (L, n_features).
        Raises:
            ValueError: If the number of centers exceeds the number of data points.
        """
        if self.L > x.shape[0]:
            raise ValueError(
                "The number of centers should be "
                "less than the number of data points. "
                f"Got: L={self.L}, N={x.shape[0]}."
            )
        x = np.asarray(x)
        rng = np.random.default_rng(seed)
        indices = rng.choice(x.shape[0], size=self.L, replace=False)
        self._centers = x[indices]
        return self._centers

    def construct_basis(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Construct the RBF basis matrix for the given input data.
        Args:
            x (npt.ArrayLike): Input data of shape (n_samples, n_features).
        Returns:
            npt.ArrayLike: Basis matrix of shape (n_samples, L).
        """
        x = np.asarray(x)
        # If only one data point is given, turn it into a 2D array.
        if x.ndim == 1:
            x = x[None, :]
        # Only compute centers once.
        if self._centers is None:
            self._centers = self.get_centers(x)
        return utils.rbf(x, self._centers, self.eps)

    def get_params(self, deep=True):
        return super().get_params(deep) | {"L": self.L, "eps": self.eps}

    def set_params(self, **params):
        self.L = params["L"]
        self.eps = params["eps"]
        return super().set_params(**params)
