"""
When using tslearn keep in mind the following convention:
Timeseries datasets are defined in R(n_ts, sz, d), where:

* n_ts is the number of timeseries in that same array/dataset
* sz is the number of timestamps defined for that dataset. For the exercises of this notebook, the timestep
that represents each observation is considered constant.
* d representing the number of dimension/variables that the timeseries describes.


"""
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class TimeSeries:
    sz: int
    d: int
    method: str | list[str]


def read_ui_har():
    directory = "./UCI HAR Dataset"
    test_data = []
    with open(f"{directory}/test/X_test.txt") as file:
        for f in file:
            test_data.append(np.fromiter(map(float, f.split()), np.float64))
    with open(f"{directory}/features.txt") as file:
        features = file.readlines()
    return test_data


class TimeSeriesGenerator:
    def __init__(
        self, drift: Callable, diffusion: Callable, initial_value, time_step, num_steps
    ):
        """
        Initialize the TimeSeriesGenerator.

        Parameters:
            drift : Callable
                A function that defines the drift of the SDE.
            diffusion : Callable
                A function that defines the diffusion of the SDE.
            initial_value : float
                The initial value of the time series.
            time_step : float
                The time step for the Euler-Maruyama method.
            num_steps : int
                The number of steps to generate in the time series.
        """
        self.drift = drift
        self.diffusion = diffusion
        self.initial_value = initial_value
        self.time_step = time_step
        self.num_steps = num_steps

    def generate(self):
        """
        Generate a time series using the Euler-Maruyama method.

        Returns:
            ndarray: An array containing the generated time series.
        """
        time_series = np.zeros(self.num_steps)
        time_series[0] = self.initial_value

        for i in range(1, self.num_steps):
            drift_value = self.drift(time_series[i - 1])
            diffusion_value = self.diffusion(time_series[i - 1])
            delta_w = np.random.normal(0, np.sqrt(self.time_step))
            time_series[i] = (
                time_series[i - 1]
                + drift_value * self.time_step
                + diffusion_value * delta_w
            )

        return time_series


# Example usage:
if __name__ == "__main__":
    # Define drift and diffusion functions (example: geometric Brownian motion)
    def drift_func(x):
        return 0.1 * x

    def diffusion_func(x):
        return 0.2 * x

    # Initialize the TimeSeriesGenerator
    generator = TimeSeriesGenerator(
        drift=drift_func,
        diffusion=diffusion_func,
        initial_value=100.0,
        time_step=0.01,
        num_steps=100,
    )

    # Generate a time series
    series_1 = generator.generate()

    # Print the first few values of the generated time series
    print(series_1[:10])
