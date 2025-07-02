import numpy as np
import numpy.typing as npt
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV
from scipy.spatial import distance_matrix


def rbf(x: npt.ArrayLike, centers: npt.ArrayLike, eps: float) -> npt.ArrayLike:
    """Computes the radial basis functions for the provided set of inputs.

    Parameters:
    -----------
    x : npt.ArrayLike, shape (N, d)
        N input points.
    centers : npt.ArrayLike, shape (M, d)
        M centers of the radial basis functions.
    eps : float
        The scaling factor of the radial basis functions.

    Returns:
    --------
    npt.Arraylike, shape (N, M)
        Evaluations of the radial basis functions.
    """
    x = np.asarray(x)
    centers = np.asarray(centers)
    if x.ndim == 1:
        x = x.reshape(1,-1)
    if centers.ndim == 1:
        centers = centers.reshape(1,-1)

    return np.exp(-(distance_matrix(x, centers))**2 / eps**2)


def linear_solve(x: npt.ArrayLike, y: npt.ArrayLike, rcond: float) -> npt.ArrayLike:
    """Computes the least-square solution for the given input and target values.

    Parameters:
    -----------
    x : npt.ArrayLike, shape (N, d_in)
        Feature matrix.
    y : npt.ArrayLike, shape (N, d_out)
        Target values.
    rcond : float
        Regualarization for the solver.

    Returns:
    --------
    npt.Arraylike, shape (d_in, d_out)
        Solution of the least-square problem.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    solution = np.linalg.lstsq(x, y, rcond=rcond)[0]
    return solution


def compute_mse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Computes the mean squared error between the true and predicted values.

    Parameters:
    -----------
    y_true : npt.ArrayLike, shape (N, d_out)
        True target values.
    y_pred : npt.ArrayLike, shape (N, d_out)
        Predicted target values.

    Returns:
    --------
    float
        Mean squared error between the true and predicted values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = np.sum((y_true - y_pred) ** 2, axis=1)
    return np.mean(errors)


def grid_search(
    parameters: dict[str, list],
    model: RegressorMixin,
    data: tuple[npt.ArrayLike, npt.ArrayLike],
    scoring: str = "neg_mean_squared_error",
    n_splits: int = 5,
    **kwargs,
) -> GridSearchCV:
    """Performs a grid search using sklearn.model_selection.GridSearchCV.

    After fitting, the best parameters and the best model can be accessed
    with cv.best_params_ and cv.best_estimator_.

    Parameters:
    -----------
        parameters : dict[str, list]
            Dictionary of possible hyperparameters for the model.
        model : RegressorMixin
            Model to be evaluated. The model should have parameters as attributes.
        data, (x, y) : tuple[npt.ArrayLike, npt.ArrayLike]
            Training data for the model.
        scoring : str, default="accuracy"
            Scoring metric for the model.
        n_splits : int, default=5
            Number of splits for cross-validation.
        **kwargs:
            Additional keyword arguments to be passed to model.fit(...).

    Returns:
    --------
    GridSearchCV
        GridSearchCV object with the results of the grid search.
    """
    # Optional: Implement the grid search with GridSearchCV.
