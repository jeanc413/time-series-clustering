import functools
from numbers import Number
from operator import itemgetter
from typing import Generator, Literal, Any
from warnings import warn

import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import (
    cdist_soft_dtw,
    cdist_soft_dtw_normalized,
    cdist_gak,
    cdist_dtw,
    cdist_ctw,
    cdist_sax,
)

from barycenter import soft_dtw_barycenter, euclidean_barycenter
from utils import c_euclidean, euclidean_distance_lc, number_try_parse


class EmptyClusterError(Exception):
    """Raised when a cluster is empty.
    """

    def __init__(self, clarification: str = ""):
        super().__init__()
        self.clarification = clarification

    def __str__(self):
        return "Computed centroid lead to empty cluster" + self.clarification


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
            epsilon: Number | str = "mean",
            min_pts: int = 4,
            distance_base_kwargs: dict[str, Any] = None,
    ):
        """Class to start a run of a DBScan clustering for time series.

        Parameters
        ----------
        series_list : list[np.ndarray] | np.ndarray[np.ndarray]
                List of time series to be clustered.
        distance_base :  Literal["euclidean", "soft_dtw", "soft_dtw_normalized", "gak", "dtw", "ctw", "sax"] = "dtw"
                Name of the distance measure to be used for clustering. See the class object `implemented` for better
                references.
        epsilon : Number | str Default : "mean"
            Epsilon value that represent the distance between points to be considered connected.
        min_pts : int
            Minimum number of points to be considered a node when running the algorithm
        distance_base_kwargs : dict[str, any]
            Key word arguments to be passed to the distance measure function.
        """
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

        if not isinstance(epsilon, Number):
            self.gather_epsilon(epsilon)

        # noinspection PyTypeChecker
        self.node_definer: np.ndarray = self.cross_matrix <= self.__epsilon
        np.fill_diagonal(self.node_definer, np.False_)

        self.labels: dict = {a: None for a in range(len(series_list))}
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
        """Internal class callable that assigns the left observations as noise.

        This is automatically called whenever the min_pts or epsilon attribute are changed.

        """
        definer = self.node_definer.sum(axis=1)
        definer = definer < self.__min_pts
        for i in np.where(definer)[0]:
            self.labels[i] = -1

    def verify_epsilon(
            self, mode: Literal["min", "mean", "median"] = "min", epsilon: float = None
    ):
        """Run to verify that the current value of epsilon is less than the value from the selected mode.

        Parameters
        ----------
        mode : Literal["min", "mean", "median"] Default = "min"
            Method to contrast the epsilon value.
        epsilon : float, optional
            Value of epsilon. If None, the current specified value of this class will be used.

        Returns
        -------
        bool
            True if the test is successful, False otherwise.

        """
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
        """Suggest epsilon values from a given set of parameters

        Returns
        -------
        dict[str, Number]
            Dictionary with the suggested and current values.

        """
        flat: np.ndarray = self.cross_matrix.copy()
        np.fill_diagonal(flat, np.nan)
        return {
            "min": np.nanmin(flat),
            "mean": np.nanmean(flat),
            "media": np.nanmedian(flat),
            "max": np.nanmax(flat),
            "current": self.__epsilon,
        }

    def gather_epsilon(self, epsilon: str):
        """Used to assign an epsilon value to this class.

        Parameters
        ----------
        epsilon : str
            Method to assign epsilon value. Possible cases are "mean", "median" or passing a float parsable string
            that represent a quantile to select the epsilon value. If the latest, only values 0 < e < 1 are possible.

        """
        if _ := number_try_parse(epsilon):
            if _ <= 0 or 1 <= _:
                raise ValueError(f"Invalid quantile value: {_}. Must be between 0 and 1")
            self.__epsilon = np.quantile(self.cross_matrix.ravel(), _)
        else:
            flat: np.ndarray = self.cross_matrix.copy()
            match epsilon:
                case "mean":
                    self.__epsilon = np.nanmean(flat)
                case "median":
                    self.__epsilon = np.nanmedian(flat)

    def fit(self):
        """Starts fitting for DBScan.

        Returns
        -------
        labels : dict[str, int]
            Cluster assignment for this class series set.

        """
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
            k: int = 3,
            distance_measure: Literal["soft_dtw", "euclidean", "dtw"] = "dtw",
            state: Generator = None,
            n_init: int = 10,
            max_iterations: int = 50,
            gamma: float = None,
            tol: float = 1e-7
    ):
        """Class to start a run of a KMeans clustering for time series.

        The implementation of this class is simil to using the random method from tslearn or sklearn.

        This class is implemented with the advantage of being able to work with multidimensional time series
        sets of different lengths.

        Parameters
        ----------
        series_list : list[np.ndarray] | np.ndarray[np.ndarray]
            List of time series to be clustered.
        k : int
         Number of clusters to use for the KMeans fitting. Default is 3.
        distance_measure :  Literal["soft_dtw", "euclidean", "dtw"]
            Distance measure to use when running this algorithm. Default is "dtw"
            For more references see the `implemented` object of this class.
        state : Generator, optional
            Seed state that can be used to sample points. Default is None.
            When None, a random sample is taken.
        n_init : int, optional
            Number of times that the KMeans algorithm will be run with different initialization points. Default is 10.
        max_iterations : int, optional
            Maximum number of iterations that an iteration of kmeans will run. Default is 50.
        gamma : float, optional
            Gamma value to pass when `distance_measure` is "soft_dtw". Default is None.
        tol : float, optional
            Tolerance for inertia improvement to consider the algorithm as converged. Default is 1e-7.

        """

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
        self.n_init = n_init
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.tol = tol

        if self.gamma is not None:
            self.distance_measure = functools.partial(
                self.distance_measure, gamma=self.gamma
            )
            self.compute_barycenter = functools.partial(
                self.compute_barycenter, gamma=self.gamma
            )

        # initialize clustering parameters
        self.max_features = max(ser.shape for ser in series_list)
        self.results: list[dict] = []
        self.criteria = "Initialized"
        self.iterations = 0
        self.inertia: int | None = None
        self.clusters = np.zeros(len(series_list))
        self.clusters_old = np.zeros(len(series_list))
        self.centroids = []
        self.centroids_old = []
        self.__old_score = np.inf
        self.score = np.inf
        self.method: Literal["random", "provided", ""] = ""

    def iterate(self, initial_centroid: list[np.ndarray]):
        """Runs one iteration of K-Means clustering.

        Parameters
        ----------
        initial_centroid : list[np.ndarray]
            Centroids to be used to initialize the algorithm

        Returns
        -------
        dict
            Dictionary containing clusters assignment, centroids, inertia, number of iterations, and exit criteria
            from running K-Means clustering with the given centroids.

        """
        iterations = 0
        centroids = initial_centroid
        clusters = np.apply_along_axis(
            np.argmin,
            axis=1,
            arr=self.distance_measure(self.series_list, centroids),
        )
        inertia = np.inf
        for _ in range(self.max_iterations):
            iterations += 1
            inertia_old = inertia

            # update clusters_definitions
            distances = self.distance_measure(self.series_list, centroids)
            clusters = np.apply_along_axis(
                np.argmin,
                axis=1,
                arr=distances,
            )
            inertia = self.compute_inertia(distances, clusters)

            # update centroids
            centroids = self._compute_centroid(clusters, current_centroids=centroids)

            # check for convergence
            if np.abs(inertia_old - inertia) < self.tol:
                criteria = "Converged by inertia"
                break

        else:
            criteria = f"Reached iteration limit at iteration={iterations}"
            warn("Algorithm stopped due to exceeding max iterations.")

        return {"clusters": clusters, "centroids": centroids,
                "inertia": inertia, "iterations": iterations, "criteria": criteria}

    def fit(self, centroids: list[np.ndarray] | None = None):
        """Run K-Means to this class time series list.

        This class is able to handle 2 clustering initialization scenarios:

        * Random: Where we randomly select k time series from the series_list and run the algorithm
        n_init times, where the best results will be stored.
        * Centroids: Where centroids are provided and the results stored are those from executing one fit
        wit the use of those centroids.

        Parameters
        ----------
        centroids : list[np.ndarray] | None
            If None, then K-Means will run n_init times, using randomly selected centroids from the
            series_list. If a list of centroids it's provided, one fit will be performed and the results stored
            in this class.

        Returns
        -------
        clusters : np.ndarray
            Assignment of clusters from the best execution of the algorithm.

        """
        # initialize centroids
        if centroids is None:
            centroids = self.state.choice(
                len(self.series_list), self.k, replace=False
            )
            centroids = itemgetter(*centroids)(self.series_list)
            self.method = "random"
        else:
            self.centroids = centroids
            self.method = "provided"
            self.n_init = 0
        best_result = self.iterate(initial_centroid=centroids)
        self.results.append(best_result)
        failed = []
        for _ in range(self.n_init):
            centroids = self.state.choice(
                len(self.series_list), self.k, replace=False
            )
            centroids = itemgetter(*centroids)(self.series_list)
            try:
                self.results.append(self.iterate(initial_centroid=centroids))
                if self.results[-1]["inertia"] < best_result["inertia"]:
                    best_result = self.results[-1]
            except EmptyClusterError:
                failed.append(_)
        if failed:
            warn(f"Failed {len(failed)}/{self.n_init} iterations={failed}.")

        # unpacking results from best iteration
        self.clusters = best_result["clusters"]
        self.centroids = best_result["centroids"]
        self.inertia = best_result["inertia"]
        self.iterations = best_result["iterations"]
        self.criteria = best_result["criteria"]
        return self.clusters

    def __fit(self, centroids=None):
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
        warn("This method is deprecated and to be removed for latest version.")
        # initialize centroids
        if centroids is None:
            self.centroids = self.state.choice(
                len(self.series_list), self.k, replace=False
            )
            self.centroids = [self.series_list[i] for i in self.centroids]
        else:
            self.centroids = centroids

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
            Reference to the closest centroid (cluster where it belongs)

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

    def _compute_centroid(self, clusters_assignment: np.ndarray, current_centroids: list[np.ndarray] = None):
        """Compute the centroids from given cluster assignments.

        Parameters
        ----------
        clusters_assignment : np.ndarray
            Cluster assignments corresponding to the current series_list
        current_centroids : list[np.ndarray], optional
            Latest centroid position to use as initial point for the barycenter calculation when possible.

        Returns
        -------
        centroids : list[np.ndarray]
            Computed centroids corresponding to the current series_list using the class defined distance_measure.

        """
        if not current_centroids:
            current_centroids = [None for _ in range(self.k)]
        clusters_list = (np.where(clusters_assignment == a, 1, 0).nonzero()[0] for a in range(self.k))
        series_in_cluster = (itemgetter(*cluster)(self.series_list) for cluster in clusters_list)
        try:
            centroids = [self.compute_barycenter(cluster, init_barycenter=current_centroids[i])
                         if isinstance(cluster, tuple)
                         else cluster
                         for i, cluster in enumerate(series_in_cluster)
                         ]
        except TypeError as e:
            if "itemgetter expected 1 argument, got 0" in str(e):
                raise EmptyClusterError("Couldn't compute centroid.")
            else:
                raise e
        return centroids

    def _get_centroids(self):
        clusters_list = (
            np.where(self.clusters == a, 1, 0).nonzero()[0]
            for a in range(len(self.centroids))
        )
        series_in_cluster = (
            itemgetter(*cluster)(self.series_list) for cluster in clusters_list
        )
        # if cluster is unitary, the centroid is the cluster unit q itself
        try:
            centroids = [
                self.compute_barycenter(clusters)
                if isinstance(clusters, tuple)
                else clusters
                for clusters in series_in_cluster
            ]
        except TypeError as e:
            if "itemgetter expected 1 argument, got 0" in str(e):
                raise EmptyClusterError("Couldn't compute centroid.")
            else:
                raise e
        return centroids

    @staticmethod
    def compute_inertia(distances: np.ndarray, assignments: np.ndarray, squared: bool = True):
        inertia = distances[np.arange(len(distances)), assignments]
        if squared:
            inertia = inertia ** 2
        return np.sum(inertia) / len(distances)

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
