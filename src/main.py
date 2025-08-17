import cProfile
from pstats import Stats, SortKey
import os
import numpy as np

from inversion.data import SyntheticData
from inversion.model_params import DispersionCurveParams
from inversion.inversion import Inversion

from plotting.plot_dispersion_curve import *

import xarray as xr


def run_inversion():
    """
    run tests for:
    - sample prior; uniform
    - sample prior; cauchy
    - start from true; low noise
    - start from true; high noise
    - generate true model; low noise
    - generate true model; high noise
    """

    sample_prior = False
    proposal_distribution = "cauchy"
    set_starting_model = False

    noise = 0.05  # real noise added to synthetic data (percentage)
    sigma_data = 0.05  # assumed noise used in likelihood calculation (percentage)
    posterior_width = {
        "depth": 0.1,
        "vel_s": 0.1,
    }  # fractional step size (multiplied by param bounds width)

    # set up data and inversion params
    bounds = {
        "depth": [0.001, 0.3],  # km
        "vel_s": [0.1, 1.8],  # km/s
    }
    model_params_kwargs = {
        "n_layers": 2,
        "vpvs_ratio": 1.75,
        "param_bounds": bounds,
        "posterior_width": posterior_width,
    }
    inversion_init_kwargs = {
        "n_burn": 5000,
        "n_chunk": 500,
        "n_mcmc": 20000,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
    }
    inversion_run_kwargs = {
        "proposal_distribution": proposal_distribution,
        "rotate_params": True,
    }

    # model params
    model_params = DispersionCurveParams(**model_params_kwargs)

    # setup data
    n_data = 50
    periods = np.flip(1 / np.logspace(0, 1.1, n_data))

    depth = [0.02, 0.04]
    vel_s = [0.2, 0.6, 1.0]

    data = SyntheticData(
        periods,
        noise,
        model_params,
        depth=depth,
        vel_s=vel_s,
    )
    model_kwargs = {"sigma_data": sigma_data * data.data_obs}

    # run inversion
    inversion = Inversion(
        data,
        model_params,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    if set_starting_model:
        # *** move to inversion ***
        # set initial model to true model
        model = inversion.chains[0]
        test_model_params = np.concatenate((depth, vel_s))

        # set initial likelihood
        model.logL, model.data_pred, model.model_params.model_params = (
            model.get_likelihood(test_model_params, data)
        )

    inversion.random_walk(
        model_params,
        **inversion_run_kwargs,
    )


def plot_inversion(file_name):

    input_path = "./results/inversion/tests/input-" + file_name + ".nc"
    results_path = "./results/inversion/tests/results-" + file_name + ".nc"

    input_ds = xr.open_dataset(input_path)
    results_ds = xr.open_dataset(results_path)

    # print(input_ds)
    # print(results_ds)

    plot_covariance_matrix(input_ds, results_ds)
    model_params_timeseries(input_ds, results_ds)
    model_params_histogram(input_ds, results_ds)
    resulting_model_histogram(input_ds, results_ds)
    plot_data_pred_histogram(input_ds, results_ds)


if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """

    # run_inversion()

    file_name = "rotation"
    plot_inversion(file_name)
