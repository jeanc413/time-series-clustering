"""
When using tslearn keep in mind the following convention:
Timeseries datasets are defined in R(n_ts, sz, d), where:

* n_ts is the number of timeseries in that same array/dataset
* sz is the number of timestamps defined for that dataset. For the exercises of this notebook, the timestep
that represents each observation is considered constant.
* d representing the number of dimension/variables that the timeseries describes.

When using the Euler-Maruyama method to generate time series data, the choice of drift and diffusion functions depends
on the underlying stochastic differential equation (SDE) that you want to model. The SDE typically takes the form:

$dX(t) = mu(X(t),t)dt + \sigma(X(t),t)dW(t)$

Here's an explanation of the types of functions you can use as drift ($\mu$) and diffusion ($\sigma$) in the context
of time series generation:

    Linear Drift and Constant Diffusion:
        Drift ($\mu$): A linear function of the current value of the time series or a constant.
        Diffusion ($\sigma$): A constant value.

    Example:
        Drift: $\mu(x, t) = \theta x + \beta$
        Diffusion: $\sigma(x, t) = \sigma_0$

    Geometric Brownian Motion (GBM):
        Drift ($\mu$): A constant times the current value.
        Diffusion ($\sigma$): A constant times the current value.

    Example:
        Drift: $\mu(x, t) = \theta x$
        Diffusion: $\sigma(x, t) = \sigma_0 x$

    Ornstein-Uhlenbeck Process:
        Drift ($\mu$): A function that drives the process back towards a mean value (mean-reverting).
        Diffusion ($\sigma$): A constant.

    Example:
        Drift: $\mu(x, t) = \theta(\mu_0 - x)$
        Diffusion: $\sigma(x, t) = \sigma_0$

    Custom Drift and Diffusion:
        You can define custom functions for drift and diffusion based on your specific modeling requirements.
        These functions could be based on domain knowledge or empirical data.

    Stochastic Volatility Models:
        For more complex financial time series modeling, you may use stochastic volatility models with time-varying
        parameters for both drift and diffusion. Examples include the Heston model, GARCH models,
        and other advanced volatility models.

The choice of drift and diffusion functions will depend on the characteristics of the data you want to model.
For financial time series, geometric Brownian motion is a common choice. For mean-reverting processes,
the Ornstein-Uhlenbeck process may be suitable. You may also need to calibrate these parameters based on real data
to match the behavior of the time series you are modeling.

Keep in mind that the specific functions and parameters used for drift and diffusion will vary based on your
application and the underlying SDE that you are trying to simulate.

"""
from dataclasses import dataclass, field
from typing import Callable
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt

from utils import ClusterScores

IMPLEMENTED_PREDICT_FLAGS = ["km", "kmeans"]
IMPLEMENTED_MODELS = ["km", "kmeans", "sc", "scan", "dbscan"]


@dataclass
class TimeSeries:
    """Representation of a time series.

    Parameters
    ----------
    sz : int | Callable[, int]
        Length of the time series. A callable that takes no arguments and returns an integer can be used.
        This callable can be easy to define using partial from the functools module.
        E.g. partial(np.random.default_rng().integers, low=0, high=5)()
    drift : Callable
        A function that defines the drift of the SDE.
    diffusion : Callable
        A function that defines the diffusion of the SDE.
    initial_value : float | np.ndarray | Callable
        The initial value of the time series. A callable that takes no arguments and returns an integer can be used.
    time_step : float
        The time step for the Euler-Maruyama method.

    """

    sz: int | Callable
    drift: Callable
    diffusion: Callable
    initial_value: float | np.ndarray | Callable
    time_step: float = 0.001

    def __post_init__(self):
        if isinstance(self.initial_value, np.ndarray) and self.initial_value.size == 1:
            warn(
                f"{self.initial_value.size=}, intended usage is a float. Attempting float casting."
            )
            self.initial_value = self.initial_value[0]
        if isinstance(self.initial_value, np.ndarray) and self.initial_value.ndim > 1:
            raise AttributeError(
                f"{self.initial_value.ndim=}, but must be unidimensional or a float."
            )
        if isinstance(self.initial_value, Callable) and not isinstance(
                self.initial_value(), (float, np.ndarray)
        ):
            raise AttributeError(
                f"Values generated from initial_value must be float or np.ndarray, "
                f"but it's returning {type(self.initial_value())=}"
            )

    def generate(self, multiprocess: bool = False, seed: np.random.Generator = None):
        """Generate a time series using the Euler-Maruyama method.

        Usage is based on a definition of drift and diffusion operators applied to SDEs typically of the form:
        $dX(t) = mu(X(t),t)dt + \sigma(X(t),t)dW(t)$

        Parameters
        ----------
        multiprocess : bool
            When generating a multivariate timeseries, lets you decide if the generation of the current step must be
            from the same or different process. The mean and variance of the process remain the same, but the
            generation will be independent for each timeseries dimension.
            Default is False.
        seed : np.random.Generator | None
            Numpy generator used to simulate the process. If None, a random generator will be used.
            Default is None

        Returns
        -------
        ndarray
            An array containing the generated time series.

        """
        if (multiprocess and isinstance(self.initial_value, float) or (
                isinstance(self.initial_value, Callable) and not isinstance(self.initial_value(), float)
        )
        ):
            raise AttributeError(
                "Multiprocess only works for multivariate time series.\n" ""
            )

        if not seed:
            seed = np.random.default_rng()

        size = self.sz() if isinstance(self.sz, Callable) else self.sz
        init_val = (
            self.initial_value()
            if isinstance(self.initial_value, Callable)
            else self.initial_value
        )

        time_series = (
            np.zeros(size)
            if isinstance(init_val, float)
            else np.zeros((size, init_val.size))
        )
        time_series[0] = init_val
        process_deviation = (
                self.time_step ** 0.5
        )  # time step is the variance of the process.
        for i in range(1, size):
            drift_value = self.drift(time_series[i - 1])
            diffusion_value = self.diffusion(time_series[i - 1])
            delta_w = seed.normal(
                0,
                process_deviation,
                None if not multiprocess else init_val.size,
            )
            time_series[i] = (
                    time_series[i - 1]
                    + drift_value * self.time_step
                    + diffusion_value * delta_w
            )

        return time_series


@dataclass
class TimeSeriesSet:
    """Class to generate a set of timeseries prepared to be used for a clustering experiment.

    Parameters
    ----------
    train_n_ts : int | list[int]
        Describe how many observations are to be generated as part of the training set from each given cluster.
        If passing an integer, it will be converted to a list of the same integer with length equal to that of clusters.
    clusters_definitions : list[TimeSeries]
        Representation of clusters to be generated.
    seed : np.random.Generator, optional
        Random number generator. If none is provided, a random SEED will be selected.
    test_n_ts : int | list[int], optional
        Describe how many observations are to be generated as part of the test set from each given cluster.
        If passing an integer, it will be converted to a list of the same integer with length equal to that of clusters.
        If none is passed, it will be assigned the same as train_n_ts
    train_set : list[np.ndarray], optional
        Generated observations for the training set.
    test_set : list[np.ndarray], optional
        Generated observations for the testing set.
    centroids : list[np.ndarray], optional
        Centroids list generated in the same order as clusters.
    train_labels : list[int], optional
        List of labels from the training set.
    test_labels : list[int], optional
        List of labels from the testing set.
    multiprocess : bool, default=True
        When generating a multivariate timeseries, lets you decide if the generation of the current step must be
            from the same or different process. The mean and variance of the process remain the same, but the
            generation will be independent for each timeseries dimension.
            Default is False.
    identifier : str
        Name that can be assigned to this series set for user identification.

    """

    train_n_ts: int | list[int]
    clusters_definitions: list[TimeSeries]
    seed: np.random.Generator = None
    test_n_ts: int | list[int] = None
    train_set: list[np.ndarray] = field(default_factory=list)
    test_set: list[np.ndarray] = field(default_factory=list)
    centroids: list[np.ndarray] = field(default_factory=list)
    train_labels: list[int] = field(default_factory=list)
    test_labels: list[int] = field(default_factory=list)
    multiprocess: bool = False
    identifier: str = None

    def __post_init__(self):
        if not self.seed:
            self.seed = np.random.default_rng()
        if not self.test_n_ts:
            self.test_n_ts = self.train_n_ts
        if not isinstance(self.train_n_ts, list):
            self.train_n_ts = [
                self.train_n_ts for _ in range(len(self.clusters_definitions))
            ]
        if not isinstance(self.test_n_ts, list):
            self.test_n_ts = [
                self.test_n_ts for _ in range(len(self.clusters_definitions))
            ]

        if len(self.test_n_ts) != len(self.train_n_ts) or len(self.test_n_ts) != len(
                self.clusters_definitions
        ):
            raise AttributeError(
                "Length of test and train set must be of the same length as "
                "the cluster definition test, but "
                f"{len(self.test_n_ts)=}, {len(self.train_n_ts)=}, {len(self.clusters_definitions)=}"
            )

    def detach_data(self):
        """Sets all generated data in this set to default values."""
        self.train_set = []
        self.test_set = []
        self.centroids = []
        self.train_labels = []
        self.test_labels = []

    def generate_set(self):
        """Generates the set of time series modeled on this class.

        Returns
        -------
        TimeSeriesSet
            Returns this same class, but containing testing and training data set and labels as well as predefined
            centroid from each class.

        """
        label = max(self.test_labels) if self.test_labels else 0
        for timeseries, test_length, train_length in zip(
                self.clusters_definitions, self.test_n_ts, self.train_n_ts
        ):
            self.test_set.extend(
                [
                    timeseries.generate(multiprocess=self.multiprocess, seed=self.seed)
                    for _ in range(test_length)
                ]
            )
            self.train_set.extend(
                [
                    timeseries.generate(multiprocess=self.multiprocess, seed=self.seed)
                    for _ in range(train_length)
                ]
            )
            self.centroids.append(
                timeseries.generate(multiprocess=self.multiprocess, seed=self.seed)
            )
            self.train_labels.extend([label for _ in range(train_length)])
            self.test_labels.extend([label for _ in range(test_length)])

            label += 1
        return self

    def plot_set(self, title: str = None) -> None:
        """Plots this class train and test set.

        Parameters
        ----------
        title : str, optional
            Title to be used for the whole plot. If identifier is assigned in this class it
            will be overloaded. Defaults to None

        """
        if self.identifier is not None:
            title = self.identifier

        figure, axes = plt.subplots(2)
        axes[0].plot(self.train_set)
        axes[0].set_title("Train set")

        axes[1].plot(self.test_set)
        axes[1].set_title("Test set")

        if title:
            figure.suptitle(title)
        figure.tight_layout()

        plt.show()


@dataclass
class Experiment:
    """Runs a determined experiment consisting of a list of TimeSeriesSets and clustering models returning
    a list of results from each of those runs.

    Parameters
    ----------
    series_sets: dict[str, TimeSeriesSet]
        Named time series sets that have been generated or not.
    partial_models: dict[str, Callable]
        List of models to experiment with. Note that this must be partial models that can be instantiated by passing
        only a list of time series
    results: list[dict] Optional
        This is the list storing results for the ran experiment.

    """

    series_sets: dict[str, TimeSeriesSet]
    partial_models: dict[str, Callable]
    results: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)

    def __post_init__(self):
        for name in self.partial_models:
            if not self.check_implemented(name):
                raise AttributeError(
                    f"Model {name=} didn't share any flag with available implemented models "
                    f"{IMPLEMENTED_MODELS=}"
                )

    @staticmethod
    def check_implemented_predict(model_name: str):
        """Checks if a provided model contains predict

        Check IMPLEMENTED_PREDICT_FLAGS for appropriate flags.

        Parameters
        ----------
        model_name : str
            Name of the model to be verified for predict.

        Returns
        -------
        implemented : bool
            True if the model contains a predict method, False otherwise.

        """
        implemented = True
        for flag in IMPLEMENTED_PREDICT_FLAGS:
            if flag in model_name:
                break
        else:
            implemented = False
        return implemented

    @staticmethod
    def check_implemented(model_name: str):
        """Check if the model is implemented for this experiment class.

        Check IMPLEMENTED_MODELS for appropriate flags to name your models.

        Parameters
        ----------
        model_name : str
            Name of the model.

        Returns
        -------
        bool
            True if the model name is implemented to be used in this class. False otherwise.

        """
        implemented = True
        for flag in IMPLEMENTED_MODELS:
            if flag in model_name:
                break
        else:
            implemented = False
        return implemented

    def run_models(self, series_name: str, series_set: TimeSeriesSet):
        """Takes a TimeSeriesSets and runs a user defined list of models.

        Parameters
        ----------
        series_name : str
            Name of the series set.
        series_set : TimeSeriesSet
            Times series set object.

        Returns
        -------
        results : list[dict]
            Results from applying the defined models in this class.
            For implemented metrics see ClusterScores.

        """
        results = []
        for alg_name, alg_model in self.partial_models.items():
            if any(
                    alg_name == r["alg_model"] and series_name == r["series_name"]
                    for r in self.results
            ):
                continue
            try:
                alg = (
                    alg_model(
                        series_list=series_set.train_set,
                        k=len(series_set.clusters_definitions),
                    )
                    if (implemented_predict := self.check_implemented_predict(alg_name))
                    else alg_model(series_list=series_set.train_set)
                )
                true_labels = series_set.train_labels

                predict_labels = (
                    alg.fit(series_set.centroids)
                    if "centroids" in dir(alg)
                    else alg.fit()
                )

                if isinstance(predict_labels, dict):
                    predict_labels = list(predict_labels.values())

                if implemented_predict:
                    true_labels = series_set.test_labels
                    predict_labels = alg.predict(series_set.test_set)
                results.append(
                    {
                        "series_name": series_name,
                        "alg_model": alg_model,
                        "iterations": alg.iterations if "iterations" in dir(alg) else None,
                        "results": alg.results if "results" in dir(alg) else None,
                        **ClusterScores(true_labels, predict_labels).get_scores(),
                    }
                )
            except Exception as e:
                self.failed.append(
                    {
                        "series_name": series_name,
                        "alg_model": alg_model,
                        "exception": e,
                    }
                )

        return results

    def run_experiment(self, detach_data: bool = True):
        """Takes all TimeSeriesSets stored in this class and runs them through the user models defined in this class.

        Parameters
        ----------
        detach_data : bool
            Whether to detach the data from the TimeSeriesSet after all models have been trained.
            This is intended to save memory

        Returns
        -------
        results : list[dict]
            Results of cross running all models and series sets in this class.

        """
        for series_name, series_set in self.series_sets.items():
            if not series_set.train_set:
                series_set: TimeSeriesSet = series_set.generate_set()
            self.results.extend(self.run_models(series_name, series_set))
            if detach_data:
                series_set.detach_data()

        if self.failed:
            warn(
                f"While running current experiment, a total of {len(self.failed)} where raised.\n"
                f"See failed argument for more details."
            )
        return self.results


# Example usage:
if __name__ == "__main__":
    # Define drift and diffusion functions (example: geometric Brownian motion)
    def drift_func(x):
        return 0.1 * x


    def diffusion_func(x):
        return 0.2 * x


    seed_number = 123

    generator = TimeSeries(
        sz=100,
        drift=drift_func,
        diffusion=diffusion_func,
        initial_value=np.array([100.0, 3.2, 43]),
        time_step=0.01,
    )
    series_1 = TimeSeries(
        sz=100,
        drift=drift_func,
        diffusion=diffusion_func,
        initial_value=np.array([100.0, 3.2, 43]),
        time_step=0.01,
    ).generate(seed=np.random.default_rng(seed_number))

    series_2 = TimeSeries(
        sz=100,
        drift=drift_func,
        diffusion=diffusion_func,
        initial_value=np.array([100.0, 3.2, 43]),
        time_step=0.01,
    ).generate(multiprocess=True, seed=np.random.default_rng(seed_number))

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(series_1)
    ax[0].set_title("Series 1\none process")
    ax[1].plot(series_2)
    ax[1].set_title("Series 2\nmultiprocess")
    plt.show()
