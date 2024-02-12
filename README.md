# Time Series Clustering

# Description

This project is a numerical complement to the Master Thesis  `Comparing Methods for Time Series Clustering`.

# Running Experiment

It is considered appropriate always to set a new environment to test or build projects.

Once such environment is created, it is important to ensure that such environment is activated:

```commandline
venv\Scripts\activate
```

And all packages can be installed by:

```commandline
pip install -r requirements.txt
```

After this, you can run any of the configured
experiments [lc, gbm, oup, trig, mv, lc-vl, gbm-vl, oup-vl, trig-vl, mv-vl].

```commandline
py experiment.py --simulation-mode lc
```

For further details on possible configurations one can run

```commandline
py experiment.py --help
```

To gather all results and generate the whole set of results with experiments its only needed to run:

```commandline
py reduce.py
```

# Contacts
If needed further information feel free to contact me at [jeanc.fernandez@gmail.com](mailto:jeanc.fernandez@gmail.com).
