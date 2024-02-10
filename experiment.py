import argparse
import pickle
from functools import partial
from pathlib import Path
from time import perf_counter, time

import numpy as np

import generator
from clustering import DBScan, CKMeans


def linear_drift(x: np.ndarray, mu: np.ndarray, beta: np.ndarray):
    return mu * x + beta


def constant_diffusion(*_, constant: np.ndarray):
    return constant


def gbm_drift_and_diffusion(x: np.ndarray, mu: np.ndarray):
    """Geometric Brownian Motion (GBM) drift and diffusion"""
    return x * mu


def oup_drift(x: np.ndarray, theta: np.ndarray, mean: np.ndarray):
    """Ornstein-Uhlenbeck Process"""
    return theta * (mean - x)


def trigonometric_drift(x: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    """Sinusoidal drift, non-linear"""
    return np.sin(alpha * x) + np.cos(beta * x)


def multivariate_drift(x: np.ndarray,
                       a: np.ndarray, b: np.ndarray,  # linear drift
                       mu: np.ndarray,  # gbm drift
                       theta: np.ndarray, mean: np.ndarray,  # oup drift
                       alpha: np.ndarray, beta: np.ndarray  # trig drift
                       ) -> np.ndarray:
    """Multivariate drift combining linear drift, gbm, oup and trigonometric drift."""
    result = np.empty(4)
    result[0] = a * x[0] + b
    result[1] = x[1] * mu
    result[2] = theta * (mean - x[2])
    result[3] = np.sin(alpha * x[3]) + np.cos(beta * x[3])
    return result


def multivariate_diffusion(x: np.ndarray,
                           c_l: np.ndarray,  # linear diffusion
                           mu: np.ndarray,  # gbm diffusion
                           c_o: np.ndarray,  # oup diffusion
                           c_t: np.ndarray  # trig diffusion
                           ) -> np.ndarray:
    """Multivariate diffusion combining constant drift, gbm, constant and constant."""
    return np.array([c_l, mu * x[1], c_o, c_t])


if __name__ == "__main__":
    SIMULATIONS_MODES = ['lc', 'gbm', 'oup', 'trig', 'mv']
    SIMULATIONS_MODES.extend([f"{m}-vl" for m in SIMULATIONS_MODES])

    parser = argparse.ArgumentParser(
        description="Runs an experiment to determine clustering performance vs specified data.")
    parser.add_argument("--mode")
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--train-size', type=int, default=30)
    parser.add_argument('--test-size', type=int, default=15)
    parser.add_argument('--series-length', type=int, default=100)
    parser.add_argument('--num-sets', type=int, default=30)
    parser.add_argument('--num-clusters', type=int, default=3)
    parser.add_argument('--normal-loc', type=float, default=150.)
    parser.add_argument('--normal-scale', type=float, default=80.)
    parser.add_argument('--simulation-mode', type=str, choices=SIMULATIONS_MODES)

    args = parser.parse_args()

    SEED = args.seed
    DB_SCAN_EPSILON_METHOD = "0.33"
    TRAIN_SET_SIZE = args.train_size
    TEST_SET_SIZE = args.test_size
    SERIES_LENGTH = args.series_length
    NUMBER_OF_SETS = args.num_sets
    NUMBER_OF_CENTROIDS = args.num_clusters

    len_seed = np.random.default_rng(SEED)
    len_gen = partial(len_seed.integers, low=45, high=55)
    n_gen = partial(np.random.default_rng(SEED).normal, loc=args.normal_loc, scale=args.normal_scale)

    experiment_seed = np.random.default_rng(SEED)
    STANDARD_SERIES = partial(generator.TimeSeries, sz=SERIES_LENGTH)
    STANDARD_MODEL = partial(generator.TimeSeriesSet, train_n_ts=TRAIN_SET_SIZE, test_n_ts=TEST_SET_SIZE,
                             seed=experiment_seed)

    match args.simulation_mode:
        case 'lc':
            simulation_set = {f"Linear Drift with Constant Diffusion {i}": STANDARD_MODEL(
                clusters_definitions=[STANDARD_SERIES(
                    drift=partial(linear_drift, mu=n_gen(), beta=n_gen()),
                    diffusion=partial(constant_diffusion, constant=n_gen()),
                    initial_value=n_gen()
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Linear Drift with Constant Diffusion")
                for i in range(NUMBER_OF_SETS)}
        case "lc-vl":
            simulation_set = {f"Linear Drift with Constant Diffusion variable length {i}": STANDARD_MODEL(
                clusters_definitions=[generator.TimeSeries(
                    sz=len_gen(),
                    drift=partial(linear_drift, mu=n_gen(), beta=n_gen()),
                    diffusion=partial(constant_diffusion, constant=n_gen()),
                    initial_value=n_gen()
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Linear Drift with Constant Diffusion variable length")
                for i in range(NUMBER_OF_SETS)}

        case "gbm":
            simulation_set = {f"Geometric Brownian Motion {i}": STANDARD_MODEL(
                clusters_definitions=[STANDARD_SERIES(
                    drift=partial(gbm_drift_and_diffusion, mu=n_gen() / args.normal_loc),  # drift is percentual
                    diffusion=partial(gbm_drift_and_diffusion, mu=n_gen() / args.normal_loc),  # diffusion is percentual
                    initial_value=n_gen()
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Geometric Brownian Motion")
                for i in range(NUMBER_OF_SETS)}
        case "gbm-vl":
            simulation_set = {f"Geometric Brownian Motion variable length {i}": STANDARD_MODEL(
                clusters_definitions=[generator.TimeSeries(
                    sz=len_gen(),
                    drift=partial(gbm_drift_and_diffusion, mu=n_gen() / args.normal_loc),
                    diffusion=partial(gbm_drift_and_diffusion, mu=n_gen() / args.normal_loc),
                    initial_value=n_gen()
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Geometric Brownian Motion variable length")
                for i in range(NUMBER_OF_SETS)}

        case "oup":
            simulation_set = {f"Ornstein-Uhlenbeck Process {i}": STANDARD_MODEL(
                clusters_definitions=[STANDARD_SERIES(
                    drift=partial(oup_drift, theta=n_gen(), mean=n_gen()),
                    diffusion=partial(constant_diffusion, constant=n_gen()),
                    initial_value=n_gen()
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Ornstein-Uhlenbeck Process")
                for i in range(NUMBER_OF_SETS)}
        case "oup-vl":
            simulation_set = {f"Ornstein-Uhlenbeck Process variable length {i}": STANDARD_MODEL(
                clusters_definitions=[generator.TimeSeries(
                    sz=len_gen(),
                    drift=partial(oup_drift, theta=n_gen(), mean=n_gen()),
                    diffusion=partial(constant_diffusion, constant=n_gen()),
                    initial_value=n_gen()
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Ornstein-Uhlenbeck Process variable length")
                for i in range(NUMBER_OF_SETS)}

        case "trig":
            simulation_set = {f"Trigonometric non-linear {i}": STANDARD_MODEL(
                clusters_definitions=[STANDARD_SERIES(
                    drift=partial(trigonometric_drift, alpha=n_gen(), beta=n_gen()),
                    diffusion=partial(constant_diffusion, constant=n_gen()),
                    initial_value=n_gen()
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Trigonometric non-linear")
                for i in range(NUMBER_OF_SETS)}
        case "trig-vl":
            simulation_set = {f"Trigonometric non-linear variable length {i}": STANDARD_MODEL(
                clusters_definitions=[generator.TimeSeries(
                    sz=len_gen(),
                    drift=partial(trigonometric_drift, alpha=n_gen(), beta=n_gen()),
                    diffusion=partial(constant_diffusion, constant=n_gen()),
                    initial_value=n_gen()
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Trigonometric non-linear variable length")
                for i in range(NUMBER_OF_SETS)}

        case "mv":
            simulation_set = {f"Multivariate {i}": STANDARD_MODEL(
                clusters_definitions=[STANDARD_SERIES(
                    drift=partial(multivariate_drift, a=n_gen(), b=n_gen(), mu=n_gen() / args.normal_loc, theta=n_gen(),
                                  mean=n_gen(),
                                  alpha=n_gen(), beta=n_gen()),
                    diffusion=partial(multivariate_diffusion, c_l=n_gen(), mu=n_gen() / args.normal_loc, c_o=n_gen(),
                                      c_t=n_gen()),
                    initial_value=n_gen(size=4)
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Multivariate")
                for i in range(NUMBER_OF_SETS)}
        case "mv-vl":
            simulation_set = {f"Multivariate variable length {i}": STANDARD_MODEL(
                clusters_definitions=[generator.TimeSeries(
                    sz=len_gen(),
                    drift=partial(multivariate_drift, a=n_gen(), b=n_gen(), mu=n_gen() / args.normal_loc,
                                  theta=n_gen(), mean=n_gen(), alpha=n_gen(), beta=n_gen()),
                    diffusion=partial(multivariate_diffusion, c_l=n_gen(), mu=n_gen() / args.normal_loc, c_o=n_gen(),
                                      c_t=n_gen()),
                    initial_value=n_gen(size=4)
                ) for _ in range(NUMBER_OF_CENTROIDS)],
                identifier="Multivariate variable length")
                for i in range(NUMBER_OF_SETS)}

        case _:
            raise AttributeError(
                f"--simulation-mode={args.simulation_mode} not supported. Must be in {SIMULATIONS_MODES}")

    experiment = generator.Experiment(
        series_sets=simulation_set,
        partial_models={
            "km-euclidean": partial(CKMeans, distance_measure="euclidean", state=experiment_seed),
            "km-dtw": partial(CKMeans, distance_measure="dtw", state=experiment_seed),
            "km-soft_dtw-g=0.5": partial(CKMeans, distance_measure="soft_dtw", state=experiment_seed, gamma=0.5),
            "km-soft_dtw-g=1": partial(CKMeans, distance_measure="soft_dtw", state=experiment_seed, gamma=1.0),
            "km-soft_dtw-g=2": partial(CKMeans, distance_measure="soft_dtw", state=experiment_seed, gamma=2.0),
            "km-soft_dtw-g=3": partial(CKMeans, distance_measure="soft_dtw", state=experiment_seed, gamma=3.0),
            "sc-euclidean": partial(DBScan, distance_base="euclidean", epsilon=DB_SCAN_EPSILON_METHOD, min_pts=6),
            "sc-dtw": partial(DBScan, distance_base="dtw", epsilon=DB_SCAN_EPSILON_METHOD, min_pts=6),
            "sc-soft_dtw-g=0.5": partial(DBScan, distance_base="soft_dtw", epsilon=DB_SCAN_EPSILON_METHOD, min_pts=6,
                                         distance_base_kwargs={"gamma": 0.5},
                                         ),
            "sc-soft_dtw-g=1": partial(DBScan, distance_base="soft_dtw", epsilon=DB_SCAN_EPSILON_METHOD, min_pts=6,
                                       distance_base_kwargs={"gamma": 1},
                                       ),
            "sc-soft_dtw-g=2": partial(DBScan, distance_base="soft_dtw", epsilon=DB_SCAN_EPSILON_METHOD, min_pts=6,
                                       distance_base_kwargs={"gamma": 2},
                                       ),
            "sc-soft_dtw-g=3": partial(DBScan, distance_base="soft_dtw", epsilon=DB_SCAN_EPSILON_METHOD, min_pts=6,

                                       distance_base_kwargs={"gamma": 3},
                                       ),
        },
    )

    t_0 = time()
    p_0 = perf_counter()
    results = experiment.run_experiment()
    p_f = perf_counter()
    t_f = time()
    print(f"Processing time of experiment {args.simulation_mode} lasted {t_f - t_0}.")
    print(f"Performance counter of experiment {args.simulation_mode} lasted {p_f - p_0}.")
    print("")

    Path('./experiments-results').mkdir(exist_ok=True)
    with open(f"./experiments-results/{args.simulation_mode}.experiment", 'wb') as file:
        pickle.dump(experiment, file)
