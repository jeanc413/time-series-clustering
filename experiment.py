import argparse
import pickle
from functools import partial
from pathlib import Path
from time import perf_counter

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


SIMULATIONS_MODES = ['lc', 'gbm', 'oup', 'trig']
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
parser.add_argument('--num-sets', type=int, default=15)
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
    case str(name) if "lc" in name:
        simulation_set = {f"L-Drift|C-Diffusion {i}": STANDARD_MODEL(
            clusters_definitions=[STANDARD_SERIES(
                drift=partial(linear_drift, mu=n_gen(), beta=n_gen()),
                diffusion=partial(constant_diffusion, constant=n_gen()),
                initial_value=n_gen()
            ) for _ in range(NUMBER_OF_CENTROIDS)],
            identifier="Linear Drift with Constant Diffusion")
            for i in range(NUMBER_OF_SETS)}
    case str(name) if "lc-vl" in name:
        simulation_set = {f"L-Drift|C-Diffusion variable length {i}": STANDARD_MODEL(
            clusters_definitions=[generator.TimeSeries(
                sz=len_gen(),
                drift=partial(linear_drift, mu=n_gen(), beta=n_gen()),
                diffusion=partial(constant_diffusion, constant=n_gen()),
                initial_value=n_gen()
            ) for _ in range(NUMBER_OF_CENTROIDS)],
            identifier="L-Drift|C-Diffusion variable length")
            for i in range(NUMBER_OF_SETS)}

    case str(name) if "gbm" in name:
        simulation_set = {f"Geometric Brownian Motion {i}": STANDARD_MODEL(
            clusters_definitions=[STANDARD_SERIES(
                drift=partial(gbm_drift_and_diffusion, mu=n_gen() / 10),
                diffusion=partial(gbm_drift_and_diffusion, mu=n_gen() / 10),
                initial_value=n_gen()
            ) for _ in range(NUMBER_OF_CENTROIDS)],
            identifier="Geometric Brownian Motion")
            for i in range(NUMBER_OF_SETS)}
    case str(name) if "gbm-vl" in name:
        simulation_set = {f"L-Drift|C-Diffusion variable length {i}": STANDARD_MODEL(
            clusters_definitions=[generator.TimeSeries(
                sz=len_gen(),
                drift=partial(gbm_drift_and_diffusion, mu=n_gen() / 10),
                diffusion=partial(gbm_drift_and_diffusion, mu=n_gen() / 10),
                initial_value=n_gen()
            ) for _ in range(NUMBER_OF_CENTROIDS)],
            identifier="L-Drift|C-Diffusion variable length")
            for i in range(NUMBER_OF_SETS)}

    case str(name) if "oup" in name:
        simulation_set = {f"Ornstein-Uhlenbeck Process {i}": STANDARD_MODEL(
            clusters_definitions=[STANDARD_SERIES(
                drift=partial(oup_drift, theta=n_gen(), mean=n_gen()),
                diffusion=partial(constant_diffusion, constant=n_gen()),
                initial_value=n_gen()
            ) for _ in range(NUMBER_OF_CENTROIDS)],
            identifier="Ornstein-Uhlenbeck Process")
            for i in range(NUMBER_OF_SETS)}
    case str(name) if "oup-vl" in name:
        simulation_set = {f"L-Drift|C-Diffusion variable length {i}": STANDARD_MODEL(
            clusters_definitions=[generator.TimeSeries(
                sz=len_gen(),
                drift=partial(oup_drift, theta=n_gen(), mean=n_gen()),
                diffusion=partial(constant_diffusion, constant=n_gen()),
                initial_value=n_gen()
            ) for _ in range(NUMBER_OF_CENTROIDS)],
            identifier="L-Drift|C-Diffusion variable length")
            for i in range(NUMBER_OF_SETS)}

    case str(name) if "trig" in name:
        simulation_set = {f"Trigonometric non-linear {i}": STANDARD_MODEL(
            clusters_definitions=[STANDARD_SERIES(
                drift=partial(trigonometric_drift, alpha=n_gen(), beta=n_gen()),
                diffusion=partial(constant_diffusion, constant=n_gen()),
                initial_value=n_gen()
            ) for _ in range(NUMBER_OF_CENTROIDS)],
            identifier="Trigonometric non-linear")
            for i in range(NUMBER_OF_SETS)}
    case str(name) if "trig-vl" in name:
        simulation_set = {f"Trigonometric non-linear variable length {i}": STANDARD_MODEL(
            clusters_definitions=[generator.TimeSeries(
                sz=len_gen(),
                drift=partial(trigonometric_drift, alpha=n_gen(), beta=n_gen()),
                diffusion=partial(constant_diffusion, constant=n_gen()),
                initial_value=n_gen()
            ) for _ in range(NUMBER_OF_CENTROIDS)])
            for i in range(NUMBER_OF_SETS)}

    case _:
        raise AttributeError(
            f"Provided --simulation-mode={args.simulation_mode} is not supported. Must be inside {SIMULATIONS_MODES}")

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

t_0 = perf_counter()
results = experiment.run_experiment()
t_f = perf_counter()
print(f"Processing time of experiment {args.simulation_mode} lasted {t_f - t_0}")
Path('./experiments-results').mkdir(exist_ok=True)

with open(f"./experiments-results/{args.simulation_mode}.experiment", 'wb') as file:
    pickle.dump(experiment, file)
