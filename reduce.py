import pickle
from pathlib import Path

import pandas as pd

from experiment import linear_drift, constant_diffusion, gbm_drift_and_diffusion, oup_drift, trigonometric_drift
from clustering import CKMeans, DBScan


def refactor(experiment_result: dict):
    refactored_results = {"simulation_mode": ' '.join(experiment_result["series_name"].split(' ')[:-1]),
                          "algorithm": CKMeans if issubclass(experiment_result['alg_model'], CKMeans) else DBScan,
                          **experiment_result['alg_model'].keywords}

    return refactored_results


results = []
for p in Path("./experiments-results").glob("*.experiment"):
    with open(p, "rb") as file:
        experiment = pickle.load(file)
        results.extend([{**refactor(r), **r} for r in experiment.results])

# noinspection PyTypeChecker
data = pd.DataFrame(results)
data.to_excel("raw_data.xlsx")
summarized_results = (data
                      .assign(alg=lambda df: df.alg_model.apply(str))
                      .groupby(['simulation_mode', 'alg'])
                      .describe())
summarized_results.to_excel("summary.xlsx")
