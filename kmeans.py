from operator import itemgetter
import numpy as np
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from tslearn.metrics import soft_dtw

from barycenter import soft_dtw_barycenter
from typing import Callable, Iterable
from warnings import warn


def sdtw_euclidean(x: np.ndarray, y: np.ndarray, gamma: float = 0.5):
    """Computes the soft dynamic time warping measure given a certain gamma value by using Euclidean Distance
    to build the Alignment matrix

    Parameters
    ----------
    x : np.ndarray
        First timeseries to compare
    y : np.ndarray
        Second timeseries to compare
    gamma : float > 0


    Returns
    -------
    float
        Computed distance between x and y given a certain gamma value for soft dynamic time warping

    """
    if gamma < 0:
        raise AttributeError("Gamma value cannot be negative")
    return SoftDTW(SquaredEuclidean(X=x, Y=y), gamma=gamma).compute()


class KMeans:
    def __init__(
        self,
        series_list: list[np.ndarray] | np.ndarray[np.ndarray],
        k: int = (6,),
        distance_measure: Callable[[np.ndarray, np.ndarray], float] = soft_dtw,
        compute_barycenter: Callable[
            [Iterable[np.ndarray]], np.ndarray
        ] = soft_dtw_barycenter,
        max_iterations: int = 100,
    ):
        """KMeans clustering for timeseries.

        Parameters
        ----------
            series_list: list[np.ndarray] | np.ndarray[np.ndarray]
                List of SubTensors objects to be clustered.
            distance_measure: Callable[[np.ndarray, np.ndarray], float]
                Computes the distance/similarity measure between 2 timeseries.
            k: int
                Number of clusters to build.
            max_iterations: int
                Admissible iterations to compute.

        """
        # capture input parameters

        self.series_list = series_list
        self.k = k
        self.distance_measure = distance_measure
        self.compute_barycenter = compute_barycenter
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
            self.centroids = np.random.choice(
                len(self.series_list), self.k, replace=False
            )
            self.centroids = [self.series_list[i] for i in self.centroids]

        # Optimization
        for _ in range(self.max_iterations):
            self.iterations += 1
            self.centroids_old = centroids_old = self.centroids.copy()
            self.clusters_old = clusters_old = self.clusters.copy()

            # update clusters
            self.clusters = self._create_clusters(self.centroids)

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

    def _create_clusters(self, centroids):
        # Init temporal empty cluster
        clusters = [[] for _ in range(self.k)]
        # Checks for each timeseries the closest centroid and returns this as clusters list
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
