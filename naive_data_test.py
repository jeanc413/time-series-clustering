"""
When using tslearn keep in mind the following convention:
Timeseries datasets are defined in R(n_ts, sz, d), where:

* n_ts is the number of timeseries in that same array/dataset
* sz is the number of timestamps defined for that dataset. For the exercises of this notebook, the timestep
that represents each observation is considered constant.
* d representing the number of dimension/variables that the timeseries describes.


"""
import pickle
from collections import Counter
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.generators import random_walks
from tslearn.metrics import dtw, soft_dtw

import clustering
from barycenter import euclidean_barycenter, soft_dtw_barycenter
from utils import CaptureData, euclidean_distance_lc, ClusterScores

# %% generating data in random walk while capturing the non-recoverable parameters
FORCE = False
FILE_NAME = "data_and_labels.pickle"
state = np.random.default_rng(123)
mean_choices = [-2, -1, 0, 1, 2]
k = len(mean_choices)
runtimes = {}
if Path(FILE_NAME).exists() and not FORCE:
    with open(FILE_NAME, "rb") as file:
        train_data, train_labels, test_data, test_labels = pickle.load(file)
else:
    train_labels = CaptureData()
    test_labels = CaptureData()
    train_data = [
        random_walks(
            n_ts=1,
            sz=state.integers(125, 175),
            d=3,
            mu=train_labels(state.choice(mean_choices)),
            std=state.normal(1, 0.05),
        )[0]
        for _ in range(250)
    ]
    test_data = [
        random_walks(
            n_ts=1,
            sz=state.integers(125, 175),
            d=3,
            mu=test_labels(state.choice(mean_choices)),
            std=state.normal(1, 0.05),
        )[0]
        for _ in range(50)
    ]
    with open(FILE_NAME, "wb") as file:
        pickle.dump((train_data, train_labels, test_data, test_labels), file)

print(f"Train label distribution {Counter(train_labels)}")
print(f"Test label distribution {Counter(test_labels)}")

# %% euclidean
t_i = time()
alg_euclidean = clustering.KMeans(
    series_list=train_data,
    k=k,
    distance_measure=euclidean_distance_lc,
    compute_barycenter=euclidean_barycenter,
)
alg_euclidean.fit()
t_f = time()
runtimes["alg_euclidean"] = t_f - t_i
scores_euclidean = ClusterScores(
    test_labels,
    np.fromiter((alg_euclidean.predict(t) for t in test_data), np.int8, count=50),
)
scores_euclidean.print_scores()

# %% dtw, gamma = 0
t_i = time()
alg_dtw_g0 = clustering.KMeans(
    series_list=train_data,
    k=k,
    distance_measure=dtw,
    compute_barycenter=dtw_barycenter_averaging,
)
alg_dtw_g0.fit()
t_f = time()
runtimes["alg_dtw_g0"] = t_f - t_i
scores_dtw_g0 = ClusterScores(
    test_labels,
    np.fromiter((alg_dtw_g0.predict(t) for t in test_data), np.int8, count=50),
)
scores_dtw_g0.print_scores()

# %% soft-dtw, gamma = 1
t_i = time()
alg_sdtw_g1 = clustering.KMeans(series_list=train_data, k=k)
alg_sdtw_g1.fit()
t_f = time()
runtimes["alg_sdtw_g1"] = t_f - t_i
scores_sdtw_g1 = ClusterScores(
    test_labels,
    np.fromiter((alg_sdtw_g1.predict(t) for t in test_data), np.int8, count=50),
)
scores_sdtw_g1.print_scores()

# %% soft-dtw, gamma = 2
t_i = time()
alg_sdtw_g2 = clustering.KMeans(
    series_list=train_data,
    k=k,
    distance_measure=lambda s1, s2: soft_dtw(s1, s2, gamma=2),
    compute_barycenter=lambda timeseries_data: soft_dtw_barycenter(
        timeseries_data, gamma=2
    ),
)
alg_sdtw_g2.fit()
t_f = time()
runtimes["alg_sdtw_g2"] = t_f - t_i
scores_sdtw_g2 = ClusterScores(
    test_labels,
    np.fromiter((alg_sdtw_g2.predict(t) for t in test_data), np.int8, count=50),
)
scores_sdtw_g2.print_scores()

# %% summarize results
results = pd.DataFrame(
    {
        "names": ["euclidean", "dtw", "sdtw_g1", "sdtw_g2"],
        "_res": [
            s.get_scores()
            for s in (scores_euclidean, scores_dtw_g0, scores_sdtw_g1, scores_sdtw_g2)
        ],
    }
)

results = pd.concat(
    (results.drop(columns=["_res"]), pd.json_normalize(results["_res"])), axis=1
)

# %% TODO
# TODO: make cluster visualization
# TODO: make simple one or more timeseries visualization (see data)
# TODO: verify barycenter weights formulation vs paper
# TODO: define and implement a weight strategy for barycenter averaging
# TODO: improve kmeans performance (profiling)
# TODO: improve initialization of clusters (see kmeans++)
