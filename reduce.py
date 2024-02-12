import pickle
from pathlib import Path

import pandas as pd

from experiment import (linear_drift, constant_diffusion, gbm_drift_and_diffusion, oup_drift, trigonometric_drift,
                        multivariate_drift, multivariate_diffusion)
from clustering import CKMeans


def refactor(experiment_result: dict):
    refactored_results = {"simulation_mode": ' '.join(experiment_result["series_name"].split(' ')[:-1]),
                          "algorithm": "K-Means" if issubclass(experiment_result['alg_model'].func,
                                                               CKMeans) else "DBSCAN",
                          **experiment_result['alg_model'].keywords}

    return refactored_results


def try_get_gamma(gamma: dict[str, float]):
    try:
        gamma = gamma['gamma']
    except TypeError:
        pass
    return gamma


def refactor_distance_measure(frame_section: pd.DataFrame):
    distance_measure = frame_section['distance_measure'].tolist()
    distance_measure = [b if pd.isnull(m) else m for m, b in
                        zip(distance_measure, frame_section['distance_base'].tolist())]
    gamma = frame_section['gamma'].tolist()
    gamma = [try_get_gamma(b) if pd.isnull(m) else m for m, b in zip(gamma, frame_section['distance_base_kwargs'])]
    return [d if pd.isnull(g) else f'{d}, gamma={g}' for d, g in zip(distance_measure, gamma)]


results = []
for p in Path("./experiments-results").glob("*.experiment"):
    with open(p, "rb") as file:
        experiment = pickle.load(file)
        results.extend([{**refactor(r), **r} for r in experiment.results])

# noinspection PyTypeChecker
data = pd.DataFrame(results)
data.to_excel("raw_data.xlsx")
summarized_results = (data
                      .assign(algorithm=lambda df: df.algorithm.apply(str),
                              distance_measure=lambda df: refactor_distance_measure(df))
                      .drop(columns=['iterations', 'gamma', 'min_pts'])
                      .groupby(['simulation_mode', 'algorithm', 'distance_measure'])
                      .describe())
summarized_results.to_excel("summary.xlsx")
