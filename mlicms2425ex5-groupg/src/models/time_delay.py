from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.base import RegressorMixin


@dataclass
class TimeDelayEmbedding(RegressorMixin):
    """Performs time-delay embedding for multivariate time series data.

    This class computes a time-delay embedding for the input data, 
    where the delays are defined through "time_delay"

    Attributes:
    -----------
    time_delay : npt.ArrayLike | int | list[int]
        Indicates the time delays to be used for embedding. The values can be specified as:
        - An integer: Delays are [1, ..., time_delay].
        - A list of integers: Delays are explicitly defined.
        - A numpy array: Predefined delay values.
    """

    time_delay: npt.ArrayLike | int | list[int]

    def __post_init__(self):
        # We allow 'time_delay' to be int or list when
        # creating the object. But we need to transform
        # it to the np.array then.
        if isinstance(self.time_delay, int):
            self.time_delay = np.arange(1, self.time_delay + 1)
        if isinstance(self.time_delay, list):
            self.time_delay = np.asarray(self.time_delay)

        if not isinstance(self.time_delay, npt.ArrayLike):
            raise ValueError("'time_delay' should be int, list, " "or a numpy array.")

    def transform(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Computes a time-delay embedding of the provided data.

        The delays are defined by self.time_delay, and the
        last max(self.time_delay) of datapoints are discarded.

        Parameters:
        -----------
        x : npt.ArrayLike, shape (N, d)
            Input data.

        Returns:
        --------
        npt.ArrayLike, shape (N - max_delay, d + d * len(self.time_delay))
            Time-delayed data. If d > 1, the embedding is flattened
            for each data point. The first d columns of the output array
            are the original data points.
        """
        x = np.asarray(x)

        # Handle single-dimensional input
        if x.ndim == 1:
            x = x[:, np.newaxis]

        max_delay = np.max(self.time_delay)
        N, d = x.shape

        # Ensure enough data points are available for time-delay embedding
        if N <= max_delay:
            raise ValueError(
                f"Not enough data points for time-delay embedding with max_delay = {max_delay}."
            )
        
        output_dim = d + d * len(self.time_delay)
        embedded = np.zeros((N - max_delay, output_dim))

        for i in range(N - max_delay):
            original = x[i]
            delays = [x[i + t] for t in self.time_delay]
            embedded[i] = np.hstack([original] + delays)

        return embedded
