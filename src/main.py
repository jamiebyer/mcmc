import cProfile
from pstats import Stats, SortKey
import os
import numpy as np

from inversion.data import SyntheticData
from inversion.model_params import DispersionCurveParams
from inversion.inversion import Inversion

from plotting.plot_dispersion_curve import *

import xarray as xr


np.random.seed(0)


def setup_test_data(model_params, noise, depth, vel_s):
    n_data = 50
    periods = np.flip(1 / np.logspace(0, 1.1, n_data))

    # run synthetic data that uses inversion calculations for vel_p and density
    # and optionally, setting vel_p and density exactly.

    data = SyntheticData(
        periods,
        noise,
        model_params,
        depth=depth,
        vel_s=vel_s,
    )

    return data


def setup_test_model(n_layers):
    # set up example data
    proposal_width = {
        "depth": 0.05,
        "vel_s": 0.05,
    }  # fractional step size (multiplied by param bounds width)

    # set up data and inversion params
    if n_layers == 1:
        bounds = {
            "depth": np.array([0.001, 0.3]),  # km
            # "vel_s": [0.1, 1.8],  # km/s
            "vel_s": np.array([[0.100, 0.750], [0.500, 2.000]]),  # km/s
        }
    elif n_layers == 2:
        bounds = {
            "depth": np.array([0.001, 0.3]),  # km
            # "vel_s": [0.1, 1.8],  # km/s
            "vel_s": np.array([[0.100, 0.500], [0.300, 1.000], [0.750, 2.000]]),  # km/s
        }

    model_params_kwargs = {
        "n_layers": n_layers,
        "vpvs_ratio": 1.75,
        "param_bounds": bounds,
        "proposal_width": proposal_width,
    }
    # model params
    model_params = DispersionCurveParams(**model_params_kwargs)

    return model_params


def basic_inversion(n_layers, noise, sample_prior, set_starting_model):
    """
    real noise added to synthetic data (percentage)
    assumed noise used in likelihood calculation (percentage)
    """
    sigma_data = noise

    if n_layers == 1:
        # one layer
        depth = [0.02]
        vel_s = [0.4, 1.0]
    elif n_layers == 2:
        # two layers
        depth = [0.02, 0.04]
        vel_s = [0.2, 0.6, 1.0]

    model_params = setup_test_model(n_layers)
    data = setup_test_data(model_params, noise, depth, vel_s)

    inversion_init_kwargs = {
        "n_burn": 10000,
        "n_chunk": 500,
        "n_mcmc": 50000,
        "n_cov_chunk": 200,
        "n_thin": 10,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
        "set_starting_model": set_starting_model,
        # "out_filename": out_filename,
    }

    model_kwargs = {"sigma_data": sigma_data * data.data_obs}

    # run inversion
    inversion = Inversion(
        data,
        model_params,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    return inversion, model_params


def run_inversion():
    """
    - Run with sampling prior. Run with setting the starting model, run without.
    - Run with 1 layer, 2 layers.
    - Run with low noise, medium noise, high noise.
    """
    sample_prior = False
    set_starting_model = False
    rotate = True
    n_layers = 2
    noise = 0.05  # 0.02 # 0.05 # 0.1

    inversion, model_params = basic_inversion(
        n_layers=n_layers,
        noise=noise,
        sample_prior=sample_prior,
        set_starting_model=set_starting_model,
        # out_filename=out_filename,
    )
    inversion.random_walk(
        model_params,
        proposal_distribution="cauchy",
        rotate_params=rotate,
    )


#####


def plot_inversion(file_name):

    input_path = "./results/inversion/input-" + file_name + ".nc"
    results_path = "./results/inversion/results-" + file_name + ".nc"

    input_ds = xr.open_dataset(input_path)
    results_ds = xr.open_dataset(results_path)

    plot_results(input_ds, results_ds, out_filename=file_name, plot_true_model=True)

    # save_inversion_info(input_ds, results_ds, out_filename=file_name)
    # plot_covariance_matrix(input_ds, results_ds, save=True, out_filename=file_name)
    # model_params_timeseries(input_ds, results_ds, save=False, out_filename=file_name)
    # model_params_autocorrelation(
    #     input_ds, results_ds, save=False, out_filename=file_name
    # )
    # model_params_histogram(input_ds, results_ds, save=True, out_filename=file_name)
    # resulting_model_histogram(input_ds, results_ds, save=True, out_filename=file_name)
    # plot_data_pred_histogram(input_ds, results_ds, save=True, out_filename=file_name)
    # plot_likelihood(input_ds, results_ds, save=True, out_filename=file_name)


if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """

    # run_inversion()

    file_name = "1757089084"
    plot_inversion(file_name)
