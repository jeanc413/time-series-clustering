from typing import Any

import numba.typed as nbt
import numpy as np
from numba import njit, prange
from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)


def number_try_parse(obj: Any):
    """Given a certain object, tries to parse it to a float or an integer accordingly.

    If the parsing is not possible, returns False.

    Parameters
    ----------
    obj : Any
        Object to try parse to a number.

    Returns
    -------
    Number | False
        If the parsing is successful, returns the corresponding float or an integer.
        Otherwise, returns False.

    """
    try:
        if (val := float(obj)).is_integer():
            return int(obj)
        else:
            return val

    except ValueError:
        pass
    return False


@njit(nogil=True, parallel=True)
def __difference_lc(s1, s2):
    results = np.zeros(s2.shape)
    results[: len(s1)] = s1 - s2[: len(s1)]
    results[len(s1):] = s1[-1] - s2[len(s1):]
    return results


@njit(nogil=True, parallel=True)
def euclidean_distance_lc(s1: np.ndarray, s2: np.ndarray):
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

    Returns
    -------
    np.ndarray
        Normed distance between given timeseries.

    """
    if s1.ndim > 1 and s2.ndim > 1 and s1.shape[1] != s2.shape[1]:
        raise AttributeError(
            "S1 and S2 aren't in the same dimension. See Docstring for expected format definition."
        )

    if len(s1) == len(s2):
        results = s1 - s2
    elif len(s1) < len(s2):
        results = __difference_lc(s1=s1, s2=s2)
    else:
        results = __difference_lc(s1=s2, s2=s1)

    return np.linalg.norm(results)


@njit(parallel=True)
def __c_euclidean_lc(
        series_list_1: nbt.List[np.ndarray],
        series_list_2: nbt.List[np.ndarray],
):
    dists = np.zeros((len(series_list_1), len(series_list_2)), dtype=np.float64)
    for i in prange(len(series_list_1)):
        for j in prange(len(series_list_2)):
            # warning on uint to int is not an issue, Numba gets confused with the integers i, j in parallel mode.
            dists[i, j] = euclidean_distance_lc(s1=series_list_1[i], s2=series_list_2[j])
    return dists


@njit(parallel=True)
def __self_euclidean_lc(series_list: nbt.List[np.ndarray]):
    dists = np.zeros((len(series_list), len(series_list)), dtype=np.float64)
    for i in prange(len(series_list)):
        for j in prange(i):
            # warning on uint to int is not an issue, Numba gets confused with the integers i, j in parallel mode.
            dists[i, j] = euclidean_distance_lc(s1=series_list[i], s2=series_list[j])
    return dists + dists.T


def c_euclidean(
        series_list_1: list[np.ndarray] | nbt.List[np.ndarray],
        series_list_2: list[np.ndarray] | nbt.List[np.ndarray] = None,
):
    """Computes matrix of Euclidean distances from 2 given timeseries lists.
    If the second matrix is not provided, it computes a self distance squared matrix.

    This is a wrapper for numba optimized functions. Therefore, to avoid casting computing,
    it is recommended to cast lists[np.ndarray] tp numba.typed.List[np.ndarray].


    Parameters
    ----------
    series_list_1 : list[np.ndarray] | nbt.List[np.ndarray]
        First list of timeseries.
    series_list_2 : list[np.ndarray] | nbt.List[np.ndarray] (Optional)
        Second list of timeseries. Default is None. If None is provided, it will do a self distance matrix.

    Returns
    -------
    dists : np.ndarray
        Matrix containing distances between pairs of timeseries.

    """
    if isinstance(series_list_1, list):
        series_list_1 = nbt.List(np.nan_to_num(s) for s in series_list_1)
    if isinstance(series_list_2, list):
        series_list_2 = nbt.List(np.nan_to_num(s) for s in series_list_2)

    if series_list_2 is None:
        dist = __self_euclidean_lc(series_list=series_list_1)
    else:
        dist = __c_euclidean_lc(
            series_list_1=series_list_1,
            series_list_2=series_list_2,
        )

    return dist


class ClusterScores:
    """Handles the generation of scores for a set of provided true labels and predicted labels.

    Attributes
    ----------
    implemented : dict[str: func]
        Dictionary with pairs of Score names and the implemented function for computation.
        This is a class level attribute.
    true_labels: list | np.ndarray
        Labels for ground truth describing clusters_definitions.
    predicted_labels : list | np.ndarray
        Predicted labels.
    name : str, Optional
        If desired a name value can be specified for traceability reasons.

    Methods
    -------
    get_scores()
        Returns a map of `Name of Score` and the `score`
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
            predicted_labels: list | np.ndarray,
            name: str = None,
    ):
        self.true_labels = (
            true_labels
            if isinstance(true_labels, np.ndarray)
            else np.asarray(true_labels, dtype=np.int8)
        )
        self.predicted_labels = (
            predicted_labels
            if isinstance(predicted_labels, np.ndarray)
            else np.asarray(predicted_labels, dtype=np.int8)
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
            name: func(self.true_labels, self.predicted_labels)
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
        return self.implemented[metric](self.true_labels, self.predicted_labels)
