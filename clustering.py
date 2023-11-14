from operator import itemgetter
from typing import Callable, Iterable, Generator, Literal
from warnings import warn
import functools

import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging, euclidean_barycenter
from tslearn.metrics import (
    soft_dtw,
    cdist_soft_dtw,
    cdist_soft_dtw_normalized,
    cdist_gak,
    cdist_dtw,
    cdist_ctw,
    cdist_sax,
)

from barycenter import soft_dtw_barycenter
from utils import c_euclidean, euclidean_distance_lc


class KMeans:
    def __init__(
        self,
        series_list: list[np.ndarray] | np.ndarray[np.ndarray],
        k: int = (6,),
        distance_measure: Callable[[np.ndarray, np.ndarray], float] = soft_dtw,
        compute_barycenter: Callable[
            [Iterable[np.ndarray]], np.ndarray
        ] = soft_dtw_barycenter,
        state: Generator = None,
        max_iterations: int = 10,
    ):
        """KMeans clustering for timeseries.

        Parameters
        ----------
            series_list: list[np.ndarray] | np.ndarray[np.ndarray]
                List of SubTensors objects to be clustered.
            distance_measure: Callable[[np.ndarray, np.ndarray], float]
                Computes the distance/similarity measure between 2 timeseries.
            k: int
                Number of clusters_definitions to build.
            max_iterations: int
                Admissible iterations to compute.

        """
        # capture input parameters

        self.series_list = series_list
        self.k = k
        self.distance_measure = distance_measure
        self.compute_barycenter = compute_barycenter
        self.state = state if state is not None else np.random.default_rng()
        self.max_iterations = max_iterations

        # initialize clustering parameters
        self.max_features = max(ser.shape for ser in series_list)
        self.criteria = "Initialized"
        self.iterations = 0

        # list of sample indices inside each cluster
        self.clusters = [[] for _ in range(self.k)]
        self.clusters_old = [[] for _ in range(self.k)]

        # mean feature tensor for each cluster
        self.centroids = []
        self.centroids_old = []

    def fit(self, centroids=None):
        """
        Class method to execute clustering.

        Parameters
        ----------
        centroids: list or None
            List of tensors cores containing the centroid-tensors used to initialize the algorithm.
            If `None` are provided, they randomly initialized from the provided cores

        Returns
        -------
        numpy.ndarray containing the assigned cluster for each provided SubTensor object.

        """

        # initialize centroids
        if centroids is None:
            self.centroids = self.state.choice(
                len(self.series_list), self.k, replace=False
            )
            self.centroids = [self.series_list[i] for i in self.centroids]

        # Optimization
        for _ in range(self.max_iterations):
            self.iterations += 1
            self.centroids_old = centroids_old = self.centroids.copy()
            self.clusters_old = clusters_old = self.clusters.copy()

            # update clusters_definitions
            self.clusters = self._create_clusters()

            # update centroids
            self.centroids = self._get_centroids(self.clusters)

            # check for convergence
            if all(self._is_converged(centroids_old, clusters_old)):
                self.criteria = "Converged"
                break
        else:
            self.criteria = f"Reached iteration limit at iteration={self.iterations}"
            warn("Algorithm stopped due to exceeding max iterations.")

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # Puts together on an array the assigned cluster for each tensor
        labels = np.empty(len(self.series_list))
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self):
        # Init temporal empty cluster
        clusters = [[] for _ in range(self.k)]
        # Checks for each timeseries the closest centroid and returns this as clusters_definitions list
        for idx, sample in enumerate(self.series_list):
            centroids_idx = self.predict(sample)
            clusters[centroids_idx].append(idx)
        return clusters

    def predict(self, observation: np.ndarray):
        """Given an observation of the same dimension as the timeseries defined in this class,
        it will return the closest centroid position (Therefore, the cluster where this observation belongs)

        This class is used as one of the internal methods to fit data

        Parameters
        ----------
        observation : np.ndarray
            Data point to be compared to any of the defined centroids

        Returns
        -------
        int
            Representation to the closest centroid (cluster where it belongs)

        """
        # Computes the distance from the current sample to all existent centroids
        # Checks the closest centroid and returns its index
        closest_idx = np.argmin(
            [self.distance_measure(observation, point) for point in self.centroids]
        )
        return int(closest_idx)

    def _get_centroids(self, clusters_list: list[list] | list[np.ndarray]):
        series_in_cluster = (
            itemgetter(*cluster)(self.series_list) for cluster in clusters_list
        )
        # if cluster is unitary, the centroid is the cluster unit q itself
        centroids = [
            self.compute_barycenter(clusters)
            if isinstance(clusters, tuple)
            else clusters
            for clusters in series_in_cluster
        ]
        return centroids

    def _is_converged(self, centroids_old, clusters_old):
        # Verify if there's no more improvement for the current iteration and returns True as converging criteria
        yield all(a.shape == b.shape for a, b in zip(centroids_old, self.centroids))
        yield clusters_old == self.clusters
        yield all(np.allclose(a, b) for a, b in zip(centroids_old, self.centroids))


class DBScan:
    implemented = {
        "euclidean": c_euclidean,
        "soft_dtw": cdist_soft_dtw,
        "soft_dtw_normalized": cdist_soft_dtw_normalized,
        "gak": cdist_gak,
        "dtw": cdist_dtw,
        "ctw": cdist_ctw,
        "sax": cdist_sax,
    }

    def __init__(
        self,
        series_list: list[np.ndarray] | np.ndarray[np.ndarray],
        distance_base: Literal[
            "euclidean",
            "soft_dtw",
            "soft_dtw_normalized",
            "gak",
            "dtw",
            "ctw",
            "sax",
        ] = "dtw",
        epsilon: float | str = 0.5,
        min_pts: int = 5,
        distance_base_kwargs: dict = None,
    ):
        self.series_list = series_list
        self.distance_base = distance_base
        self.__epsilon = epsilon
        self.__min_pts = min_pts
        self.distance_base_kwargs = (
            distance_base_kwargs if distance_base_kwargs is not None else {}
        )

        if distance_base_kwargs:
            self.cross_matrix: np.ndarray = self.implemented[distance_base](
                series_list, **distance_base_kwargs
            )
        else:
            self.cross_matrix: np.ndarray = self.implemented[distance_base](series_list)

        # noinspection PyTypeChecker
        self.node_definer: np.ndarray = self.cross_matrix <= epsilon
        np.fill_diagonal(self.node_definer, np.False_)

        self.labels = {a: None for a in range(len(series_list))}
        self.assign_as_noise()

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.__epsilon = value
        self.node_definer = self.cross_matrix <= value
        np.fill_diagonal(self.node_definer, np.False_)
        self.assign_as_noise()

    @property
    def min_pts(self):
        return self.__min_pts

    @min_pts.setter
    def min_pts(self, val):
        self.__min_pts = val
        self.assign_as_noise()

    def assign_as_noise(self):
        definer = self.node_definer.sum(axis=1)
        definer = definer < self.__min_pts
        for i in np.where(definer)[0]:
            self.labels[i] = -1

    def verify_epsilon(
        self, mode: Literal["min", "mean", "median"] = "min", epsilon: float = None
    ):
        if epsilon is None:
            epsilon = self.epsilon
        flat: np.ndarray = self.cross_matrix.copy()
        np.fill_diagonal(flat, np.nan)
        match mode:
            case "min":
                result = np.nanmin(flat) < epsilon
            case "mean":
                result = np.nanmean(flat) < epsilon
            case "median":
                result = np.nanmedian(flat) < epsilon
            case _:
                raise AttributeError(
                    f"{mode=} is not implemented.\nSee one of [min, mean, median]."
                )
        return result

    def suggest_epsilon(self):
        flat: np.ndarray = self.cross_matrix.copy()
        np.fill_diagonal(flat, np.nan)
        return {
            "min": np.nanmin(flat),
            "mean": np.nanmean(flat),
            "media": np.nanmedian(flat),
            "max": np.nanmax(flat),
            "current": self.__epsilon,
        }

    def fit(self):
        current_label = -1
        for key, val in self.labels.items():
            if val is not None:
                continue
            current_label += 1
            seed_set_old = np.array([])
            seed_set = np.where(self.node_definer[key])[0]

            while seed_set.shape != seed_set_old.shape:
                seed_set_old = seed_set.copy()
                cores = self.node_definer[seed_set].sum(axis=0) >= self.min_pts
                cores = np.where(cores)[0]
                seed_set = np.unique(np.array(seed_set.tolist() + cores.tolist()))

            for neighbor in seed_set:
                if self.labels[neighbor] is None or self.labels[neighbor] == -1:
                    self.labels[neighbor] = current_label
        return self.labels


class CKMeans:
    implemented = {
        "soft_dtw": {"measure": cdist_soft_dtw, "barycenter": soft_dtw_barycenter},
        "euclidean": {"measure": c_euclidean, "barycenter": euclidean_barycenter},
        "dtw": {"measure": cdist_dtw, "barycenter": dtw_barycenter_averaging},
    }

    def __init__(
        self,
        series_list: list[np.ndarray] | np.ndarray[np.ndarray],
        k: int = 6,
        distance_measure: Literal["soft_dtw", "euclidean", "dtw"] = "dtw",
        state: Generator = None,
        max_iterations: int = 10,
        gamma: float = None,
    ):
        """KMeans clustering for timeseries.

        Parameters
        ----------
            series_list: list[np.ndarray] | np.ndarray[np.ndarray]
                List of SubTensors objects to be clustered.
            distance_measure: distance_measure: Literal["soft_dtw", "euclidean", "dtw"] = "dtw"
                Name of the distance measure to be used for clustering.
            k: int
                Number of clusters_definitions to build.
            max_iterations: int
                Admissible iterations to compute.

        """
        # capture input parameters
        if distance_measure != "soft_dtw" and gamma:
            raise AttributeError(
                f"Distance measure {distance_measure} cannot use gamma parameter."
            )

        self.series_list = series_list
        self.k = k
        self.mode = distance_measure
        self.distance_measure = self.implemented[distance_measure]["measure"]
        self.compute_barycenter = self.implemented[distance_measure]["barycenter"]
        self.state = state if state is not None else np.random.default_rng()
        self.max_iterations = max_iterations
        self.gamma = gamma

        if self.gamma is not None:
            self.distance_measure = functools.partial(
                self.distance_measure, gamma=self.gamma
            )
            self.compute_barycenter = functools.partial(
                self.compute_barycenter, gamma=self.gamma
            )

        # initialize clustering parameters
        self.max_features = max(ser.shape for ser in series_list)
        self.criteria = "Initialized"
        self.iterations = 0

        # list of sample indices inside each cluster
        self.clusters = np.zeros(len(series_list))
        self.clusters_old = np.zeros(len(series_list))

        # mean feature tensor for each cluster
        self.centroids = []
        self.centroids_old = []
        self.__old_score = np.inf
        self.score = np.inf

    def fit(self, centroids=None):
        """
        Class method to execute clustering.

        Parameters
        ----------
        centroids: list or None
            List of tensors cores containing the centroid-tensors used to initialize the algorithm.
            If `None` are provided, they randomly initialized from the provided cores

        Returns
        -------
        numpy.ndarray containing the assigned cluster for each provided SubTensor object.

        """

        # initialize centroids
        if centroids is None:
            self.centroids = self.state.choice(
                len(self.series_list), self.k, replace=False
            )
            self.centroids = [self.series_list[i] for i in self.centroids]

        # Optimization
        for _ in range(self.max_iterations):
            self.iterations += 1
            self.centroids_old = centroids_old = self.centroids.copy()
            self.clusters_old = clusters_old = self.clusters.copy()

            # update clusters_definitions
            self.clusters = np.apply_along_axis(
                np.argmin,
                axis=1,
                arr=self.distance_measure(self.series_list, self.centroids),
            )

            # update centroids
            self.centroids = self._get_centroids()

            # check for convergence
            if all(self._is_converged(centroids_old, clusters_old)):
                self.criteria = "Converged"
                break
        else:
            self.criteria = f"Reached iteration limit at iteration={self.iterations}"
            warn("Algorithm stopped due to exceeding max iterations.")

        return self.clusters

    def _get_cluster_labels(self, clusters):
        # Puts together on an array the assigned cluster for each tensor
        labels = np.empty(len(self.series_list))
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def predict(self, observation: np.ndarray):
        """Given an observation of the same dimension as the timeseries defined in this class,
        it will return the closest centroid position (Therefore, the cluster where this observation belongs)

        This class is used as one of the internal methods to fit data

        Parameters
        ----------
        observation : np.ndarray
            Data point to be compared to any of the defined centroids

        Returns
        -------
        int
            Representation to the closest centroid (cluster where it belongs)

        """
        # Computes the distance from the current sample to all existent centroids
        # Checks the closest centroid and returns its index
        if isinstance(observation, np.ndarray):
            closest_idx = np.argmin(
                self.distance_measure([observation], self.centroids)
            )
            closest_idx = int(closest_idx)
        else:
            closest_idx = np.apply_along_axis(
                np.argmin,
                axis=1,
                arr=self.distance_measure(observation, self.centroids),
            )
        return closest_idx

    def _get_centroids(self):
        clusters_list = (
            np.where(self.clusters == a, 1, 0).nonzero()[0]
            for a in range(len(self.centroids))
        )
        series_in_cluster = (
            itemgetter(*cluster)(self.series_list) for cluster in clusters_list
        )
        # if cluster is unitary, the centroid is the cluster unit q itself
        centroids = [
            self.compute_barycenter(clusters)
            if isinstance(clusters, tuple)
            else clusters
            for clusters in series_in_cluster
        ]
        return centroids

    def compute_score(self):
        return sum(
            euclidean_distance_lc(np.nan_to_num(a), np.nan_to_num(b))
            for a, b in zip(self.centroids, self.centroids_old)
        )

    def _is_converged(self, centroids_old, clusters_old):
        # Verify if there's no more improvement for the current iteration and returns True as converging criteria
        self.__old_score = self.score
        self.score = self.compute_score()
        yield self.__old_score <= self.score
        yield all(a.shape == b.shape for a, b in zip(centroids_old, self.centroids))
        yield np.allclose(clusters_old, self.clusters)
        yield all(np.allclose(a, b) for a, b in zip(centroids_old, self.centroids))
