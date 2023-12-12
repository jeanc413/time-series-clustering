from functools import partial

import numpy as np
import pandas as pd

import generator
import clustering


def linear_drift(x: np.ndarray, mu: np.ndarray, beta: np.ndarray):
    return mu * x + beta


def constant_diffusion(*_, constant: np.ndarray):
    return constant


def gbm_drift_and_diffusion(x: np.ndarray, mu: np.ndarray):
    return x * mu


def oup_diffusion(x: np.ndarray, theta: np.ndarray, mean: np.ndarray):
    """Ornstein-Uhlenbeck Process"""
    return theta * (mean - x)


seed = 123

len_seed = np.random.default_rng(seed)
len_gen = partial(len_seed.integers, low=45, high=55)

experiment_seed = np.random.default_rng(seed)

experiment = generator.Experiment(
    series_sets={
        "linear drift and constant diffusion": generator.TimeSeriesSet(
            train_n_ts=20,
            clusters_definitions=[
                generator.TimeSeries(
                    sz=len_gen,
                    drift=partial(
                        linear_drift,
                        mu=np.array([1.5, 2, 2.5]),
                        beta=np.array([-15, -20, -25]),
                    ),
                    diffusion=partial(
                        constant_diffusion, constant=np.array([-7.5, -10, -12.5])
                    ),
                    initial_value=np.array([100, 100, 100]),
                    time_step=0.1,
                ),
                generator.TimeSeries(
                    sz=len_gen,
                    drift=partial(
                        linear_drift,
                        mu=np.array([0.5, 0.3, 0.01]),
                        beta=np.array([-12, -18, -20]),
                    ),
                    diffusion=partial(
                        constant_diffusion, constant=np.array([-7.5, -10, -12.5])
                    ),
                    initial_value=np.array([0.5, 13.5, -0.3]),
                    time_step=0.1,
                ),
                generator.TimeSeries(
                    sz=len_gen,
                    drift=partial(
                        linear_drift,
                        mu=np.array([-0, -2, -2.5]),
                        beta=np.array([-17, -25, -30]),
                    ),
                    diffusion=partial(
                        constant_diffusion, constant=np.array([-7.5, -10, -12.5])
                    ),
                    initial_value=np.array([-200, -200, -200]),
                    time_step=0.1,
                ),
            ],
            seed=experiment_seed,
            test_n_ts=10,
        )
    },
    partial_models={
        "km-euclidean": partial(
            clustering.CKMeans,
            distance_measure="euclidean",
            state=np.random.default_rng(seed),
            max_iterations=25,
        ),
        "km-dtw": partial(
            clustering.CKMeans,
            distance_measure="dtw",
            state=np.random.default_rng(seed),
            max_iterations=25,
        ),
        "km-soft_dtw-g=0.5": partial(
            clustering.CKMeans,
            distance_measure="soft_dtw",
            state=np.random.default_rng(seed),
            max_iterations=25,
            gamma=0.5,
        ),
        "km-soft_dtw-g=1": partial(
            clustering.CKMeans,
            distance_measure="soft_dtw",
            state=np.random.default_rng(seed),
            max_iterations=25,
            gamma=1.0,
        ),
        "km-soft_dtw-g=2": partial(
            clustering.CKMeans,
            distance_measure="soft_dtw",
            state=np.random.default_rng(seed),
            max_iterations=25,
            gamma=2.0,
        ),
        "km-soft_dtw-g=3": partial(
            clustering.CKMeans,
            distance_measure="soft_dtw",
            state=np.random.default_rng(seed),
            max_iterations=25,
            gamma=3.0,
        ),
        "sc-euclidean": partial(
            clustering.DBScan,
            distance_base="euclidean",
            epsilon=1000000,
            min_pts=5,
        ),
        "sc-dtw": partial(
            clustering.DBScan,
            distance_base="dtw",
            epsilon=1000000,
            min_pts=5,
        ),
        "sc-soft_dtw-g=0.5": partial(
            clustering.DBScan,
            distance_base="soft_dtw",
            epsilon=1000000,
            min_pts=5,
            distance_base_kwargs={"gamma": 0.5},
        ),
        "sc-soft_dtw-g=1": partial(
            clustering.DBScan,
            distance_base="soft_dtw",
            epsilon=1000000,
            min_pts=5,
            distance_base_kwargs={"gamma": 1},
        ),
        "sc-soft_dtw-g=2": partial(
            clustering.DBScan,
            distance_base="soft_dtw",
            epsilon=1000000,
            min_pts=5,
            distance_base_kwargs={"gamma": 2},
        ),
        "sc-soft_dtw-g=3": partial(
            clustering.DBScan,
            distance_base="soft_dtw",
            epsilon=1000000,
            min_pts=5,
            distance_base_kwargs={"gamma": 3},
        ),
    },
)


results = pd.DataFrame.from_records(experiment.run_experiment())
