# Code for soft DTW is by Mathieu Blondel under Simplified BSD license
# This code section is taken from TSLearn barycenter module and modified for compatibility reasons.
import numpy as np
from scipy.optimize import minimize
from tslearn.metrics import SquaredEuclidean, SoftDTW
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset, check_equal_size


def euclidean_barycenter(timeseries_data, weights=None, init_barycenter=None):
    """Standard Euclidean barycenter computed from a set of time series.

    Parameters
    ----------
    timeseries_data : array-like
        Time series dataset.
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.
    init_barycenter : None
        Ignored parameter for compatibility

    Returns
    -------
    np.ndarray
        Barycenter of the provided time series dataset.

    Notes
    -----
        This method requires a dataset of equal-sized time series

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> bar = euclidean_barycenter(time_series)
    >>> bar.shape
    (4, 1)
    >>> bar
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    """
    ndim = timeseries_data[0].ndim
    timeseries_data = np.nan_to_num(to_time_series_dataset(timeseries_data))
    weights = _set_weights(weights, timeseries_data.shape[0])
    barycenter = np.average(timeseries_data, axis=0, weights=weights)
    return barycenter if ndim > 1 else barycenter.reshape((-1))


def _set_weights(w, n):
    """Provides a set of weights that fulfills the shape of a unitary vector

    Parameters
    ----------
    w : list | np.ndarray
        Weights provided for barycenter averaging. If None, equally sized weights will be assigned
    n : int

    Returns
    -------
    w : np.ndarray


    """
    if isinstance(w, list):
        w = np.array(w)
    if w is None or len(w) != n:
        w = np.ones(n)
    if not np.isclose(w.sum(), 1):
        w = w / w.sum()
    return w


def __soft_dtw_func(
        barycenter_to_eval, timeseries_data, weights, barycenter_shape, gamma
):
    # Compute objective value and grad at Z.
    barycenter_to_eval = barycenter_to_eval.reshape(barycenter_shape)
    barycenter_gradient = np.zeros_like(barycenter_to_eval)
    if barycenter_to_eval.ndim == 1:
        barycenter_gradient = barycenter_to_eval.reshape((-1, 1))
    objective = 0

    if len(timeseries_data) != len(weights):
        raise AttributeError(
            "Timeseries data and provided weights do not match in length."
        )

    for data, weight in zip(timeseries_data, weights):
        d_matrix = SquaredEuclidean(barycenter_to_eval, data)
        sdtw = SoftDTW(d_matrix, gamma=gamma)
        distance = sdtw.compute()
        barycenter_gradient_temp = d_matrix.jacobian_product(sdtw.grad())
        barycenter_gradient += weight * barycenter_gradient_temp
        objective += weight * distance

    return objective, barycenter_gradient.ravel()


def soft_dtw_barycenter(
        timeseries_data,
        gamma=1.0,
        weights=None,
        method="L-BFGS-B",
        tol=1e-3,
        max_iter=50,
        init_barycenter=None,
):
    """Compute barycenter (time series averaging) under the soft-DTW [1]
    geometry.

    Soft-DTW was originally presented in [1]_.

    Parameters
    ----------
    timeseries_data : array-like, shape=(n_ts, sz, d) | list[np.ndarray]
        Time series dataset.
    gamma: float
        Regularization parameter.
        Lower is less smoothed (closer to true DTW).
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.
    method: string
        Optimization method, passed to `scipy.optimize.minimize`.
        Default: L-BFGS.
    tol: float
        Tolerance of the method used.
    max_iter: int
        Maximum number of iterations.
    init_barycenter: array or None (default: None)
        Initial barycenter to start from for the optimization process.
        If `None`, Euclidean barycenter is used as a starting point.

    Returns
    -------
    numpy.ndarray of shape (bsz, d) where `bsz` is the size of the `init` array \
            if provided or `sz` otherwise
        Soft-DTW barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> soft_dtw_barycenter(time_series, max_iter=5)
    array([[1.25161574],
           [2.03821705],
           [3.5101956 ],
           [4.36140605]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> soft_dtw_barycenter(time_series, max_iter=5)
    array([[1.21349933],
           [1.8932251 ],
           [2.67573269],
           [3.51057026],
           [4.33645802]])

    References
    ----------
    [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.

    """
    timeseries_data_ = np.nan_to_num(to_time_series_dataset(timeseries_data))
    weights = _set_weights(weights, len(timeseries_data))

    if init_barycenter is None:
        # does a simple numpy.average using the provided weights
        if check_equal_size(timeseries_data_):
            # noinspection PyTypeChecker
            init_barycenter = euclidean_barycenter(timeseries_data_, weights=weights)
        else:
            # noinspection PyTypeChecker
            init_barycenter = euclidean_barycenter(
                TimeSeriesResampler(sz=timeseries_data_.shape[1]).fit_transform(
                    timeseries_data_
                ),
                weights=weights,
            )

    res = minimize(
        lambda a: __soft_dtw_func(
            barycenter_to_eval=a,
            timeseries_data=timeseries_data,
            weights=weights,
            barycenter_shape=init_barycenter.shape,
            gamma=gamma,
        ),
        init_barycenter.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options={"maxiter": max_iter, "disp": False},
    )
    return res.x.reshape(init_barycenter.shape)
