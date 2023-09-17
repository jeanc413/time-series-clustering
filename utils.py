from typing import Any, Iterable

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from numba import njit


class CaptureData(list):
    """After being instantiated, everytime this class is called, it will
    store a unique `observation` parameter in the self-list, and return the same value.

    Examples
    --------
    >>> capture = CaptureData()
    >>> capture(10)
    10
    >>> capture(11)
    11
    >>> capture
    [10, 11]

    >>> capture = CaptureData()
    >>> capture(1)
    1
    >>> capture(["a", "b"])
    ['a', 'b']
    >>> capture(30)
    30
    >>> capture
    [1, ['a', 'b'], 30]

    """

    def __int__(self):
        super().__init__()

    def __call__(self, observation: Any):
        """Add any object to class list and return the same value

        Parameters
        ----------
        observation : Any
            Object to be added to the list.

        Returns
        -------
        observation : Any
            Same provided object

        """
        self.append(observation)
        return observation


class VisualizeSeries:
    def __int__(self, n_rows: int = 1, n_cols: int = 1):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig, self.ax = plt.subplots(n_rows, n_cols)
        self.fig: plt.figure
        self.ax: plt.Axes

    def add_timeseries(self, timeseries: np.ndarray, row: int = None, col: int = None):
        if (self.n_rows != 1 and row is None) or (self.n_cols != 1 and col is None):
            raise AttributeError(
                "When n_col or n_row are bigger than 0, row and col attributes must be specified."
            )
        if self.n_rows != 1:
            if self.n_cols != 1:
                self.ax[row, col].plot(timeseries)
            else:
                self.ax[row].plot(timeseries)
        else:
            self.ax.plot(timeseries)

    def add_many_timeseries(self, timeseries: Iterable[np.ndarray]):
        for t in timeseries:
            self.ax.plot(t)

    def show(self):
        plt.tight_layout()
        self.fig.show()


def __difference_lc(s1, s2):
    results = np.zeros(s2.shape)
    results[: len(s1)] = s1 - s2[: len(s1)]
    results[len(s1) :] = s1[-1] - s2[len(s1) :]
    return results


@njit(nogil=True)
def euclidean_distance_lc(s1: np.ndarray, s2: np.ndarray, weights: Iterable = None):
    """Takes 2 timeseries and calculates the Euclidean distance between them.
    In case that this timeseries have different lengths, it will:
    * Calculate the difference until min(len(s1), len(s2))
    * Calculate the remaining differences for the longer timeseries, with the last value from the shorter.
    * If a list of weights is provided, it will multiply the difference vector by the weights
    * Compute the Euclidean norm for the calculated difference vector,

    Parameters
    ----------
    s1 : np.ndarray
        First timeseries.
    s2 : np.ndarray
        Second timeseries.
    weights : np.ndarray
        Weights to adjust distance computation. It must be same length as the longest timeseries.

    Returns
    -------

    """
    if s1.shape[1] != s2.shape[1]:
        raise AttributeError(
            "S1 and S2 aren't in the same dimension. See Docstring for expected format definition."
        )

    if len(s1) == len(s2):
        results = s1 - s2
    elif len(s1) < len(s2):
        results = __difference_lc(s1=s1, s2=s2)
    else:
        results = __difference_lc(s1=s2, s2=s1)

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            weights = np.asarray(weights)
        results = results * weights

    return np.linalg.norm(results)


@njit(parallel=True)
def c_euclidean_lc(series_list_1, series_list_2=None):
    # TODO: finish me
    pass


class ClusterScores:
    """Handles the generation of scores for a set of provided true labels and predicted labels.

    Attributes
    ----------
    implemented : dict[str: func]
        Dictionary with pairs of Score names and the implemented function for computation.
        This is a class level attribute.
    true_labels: list | np.ndarray
        Labels for ground truth describing clusters.
    pred_labels : list | np.ndarray
        Predicted labels.
    name : str, Optional
        If desired a name value can be specified for traceability reasons.

    Methods
    -------
    get_scores()
        Returns a dictionary of pairs of `Name of Score` and the `score`
    print_scores()
        Prints the name of the score with the calculated score
    get_specific_score(metric: str)
        Given a name of a score, it will return the calculated value for that score

    """

    implemented = {
        "Rand Index": rand_score,
        "Adjusted Rand Index": adjusted_rand_score,
        "Adjusted Mutual Info Score": adjusted_mutual_info_score,
        "Normalized Mutual Info Score": normalized_mutual_info_score,
        "Homogeneity Score": homogeneity_score,
        "Completeness Score": completeness_score,
        "V Measure": v_measure_score,
    }

    def __init__(
        self,
        true_labels: list | np.ndarray,
        pred_labels: list | np.ndarray,
        name: str = None,
    ):
        self.true_labels = (
            true_labels
            if isinstance(true_labels, np.ndarray)
            else np.asarray(true_labels, dtype=np.int8)
        )
        self.pred_labels = (
            pred_labels
            if isinstance(pred_labels, np.ndarray)
            else np.asarray(pred_labels, dtype=np.int8)
        )
        self.name = name

    def __call__(self):
        return self.get_scores()

    def get_scores(self):
        """Creates a dictionary of pairs of `Name of Score` and the `score`

        Returns
        -------
        dict
            Dictionary of pairs of `Name of Score` and the `score`

        """
        return {
            name: func(self.true_labels, self.pred_labels)
            for name, func in self.implemented.items()
        }

    def print_scores(self):
        """Prints the name of the score with the calculated score"""
        for name, result in self.get_scores().items():
            print(f"{name}=={result:.2%}")
        print()

    def get_specific_score(self, metric: str):
        """Given a name of a score, it will return the calculated value for that score.

        Parameters
        ----------
        metric : str
            Name of the score to retrieve.

        Returns
        -------
        float
            Calculated score

        """
        if metric not in self.implemented:
            raise AttributeError(
                f"Score function {metric} not implemented."
                f"The value must be one of {self.implemented.keys()}"
            )
        return self.implemented[metric](self.true_labels, self.pred_labels)
