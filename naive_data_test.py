import numpy as np
import kmeans
from collections import Counter
from tslearn.generators import random_walks
from utils import CaptureData

# %% naive approach
state = np.random.default_rng(123)
data = (
    [state.normal(10, size=(np.random.randint(10, 20), 3)) for _ in range(30)]
    + [state.normal(15, size=(np.random.randint(10, 20), 3)) for _ in range(30)]
    + [state.normal(20, size=(np.random.randint(10, 20), 3)) for _ in range(30)]
)

# %%

alg = kmeans.KMeans(series_list=data, k=3, max_iterations=500)
data_labels = alg.predict()
print(Counter(data_labels))

# %% using data in cumulative sum structure (Closer to timeseries patterns)
state = np.random.default_rng(123)
capture_labels = CaptureData()
mean_choices = [-2, -1, 0, 1, 2]
k = len(mean_choices)
data_walked = [
    random_walks(
        n_ts=1,
        sz=state.integers(100, 200),
        d=3,
        mu=capture_labels(state.choice(mean_choices)),
        std=state.normal(1, 0.05),
    )
]
