# Code for soft DTW is by Mathieu Blondel under Simplified BSD license

import numpy
import numpy as np
from scipy.optimize import minimize

from tslearn.utils import to_time_series_dataset, check_equal_size
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.metrics import SquaredEuclidean, SoftDTW

from tslearn.barycenters.euclidean import euclidean_barycenter

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


def _set_weights(w, n):
    """Provides a set of weights that fulfills the shape of a unitary vector"""
    if w is None or len(w) != n:
        w = numpy.ones(n)
    if not np.isclose(w.sum(), 1):
        w = w / w.sum()
    return w


def __soft_dtw_func(
    barycenter_to_eval, timeseries_data, weights, barycenter_shape, gamma
):
    # Compute objective value and grad at Z.
    barycenter_to_eval = np.nan_to_num(barycenter_to_eval)
    barycenter_to_eval = barycenter_to_eval.reshape(barycenter_shape)
    barycenter_gradient = numpy.zeros_like(barycenter_to_eval)
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
    barycenter=None,
):
    """Compute barycenter (time series averaging) under the soft-DTW [1]
    geometry.

    Soft-DTW was originally presented in [1]_.

    Parameters
    ----------
    timeseries_data : array-like, shape=(n_ts, sz, d)
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
    barycenter: array or None (default: None)
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
    timeseries_data_ = to_time_series_dataset(timeseries_data)
    weights = _set_weights(weights, len(timeseries_data))

    if barycenter is None:
        # does a simple numpy.average using the provided weights
        if check_equal_size(timeseries_data_):
            barycenter = euclidean_barycenter(timeseries_data_, weights=weights)
        else:
            barycenter = euclidean_barycenter(
                TimeSeriesResampler(sz=timeseries_data_.shape[1]).fit_transform(
                    timeseries_data_
                ),
                weights=weights,
            )
    barycenter_shape = barycenter.shape
    # timeseries_data = numpy.array(
    #     [to_time_series(d, remove_nans=True) for d in timeseries_data]
    # ) Unnecessary step from previous alg

    res = minimize(
        lambda a: __soft_dtw_func(
            barycenter_to_eval=a,
            timeseries_data=timeseries_data,
            weights=weights,
            barycenter_shape=barycenter_shape,
            gamma=gamma,
        ),
        barycenter.ravel(),
        method=method,
        jac=True,
        tol=tol,
        options=dict(maxiter=max_iter, disp=False),
    )
    return res.x.reshape(barycenter_shape)
