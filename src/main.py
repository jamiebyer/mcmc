import cProfile
from pstats import Stats, SortKey
import os
import numpy as np

from inversion.data import SyntheticData
from inversion.model_params import DispersionCurveParams
from inversion.inversion import Inversion

from plotting.plot_dispersion_curve import *

# from plotting.plot_dispersion_curve_plotly import *

import xarray as xr

np.random.seed(0)


def setup_test_data(model_params, noise_dist, noise_params, depth, vel_s):
    n_data = 50
    periods = np.flip(1 / np.logspace(0, 1.1, n_data))

    # run synthetic data that uses inversion calculations for vel_p and density
    # and optionally, setting vel_p and density exactly.

    data = SyntheticData(
        periods,
        noise_dist,
        noise_params,
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
            "depth": np.array([0.001, 0.15]),  # km
            # "vel_s": [0.1, 1.8],  # km/s
            "vel_s": np.array([[0.100, 0.750], [0.500, 2.000]]),  # km/s
        }
    elif n_layers == 2:
        bounds = {
            "depth": np.array([0.001, 0.15]),  # km
            # "vel_s": [0.1, 1.8],  # km/s
            "vel_s": np.array([[0.100, 0.500], [0.300, 1.000], [0.750, 2.000]]),  # km/s
        }
    elif n_layers == 3:
        bounds = {
            "depth": np.array([0.001, 0.15]),  # km
            "vel_s": np.array([0.100, 2.000]),  # km/s
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


def basic_inversion(
    n_layers,
    noise_dist,
    noise_params,
    inv_noise_dist,
    inv_noise_params,
    set_starting_model,
):
    """
    real noise added to synthetic data (percentage)
    assumed noise used in likelihood calculation (percentage)
    """

    if n_layers == 1:
        # one layer
        depth = [0.05]
        vel_s = [0.4, 1.0]
    elif n_layers == 2:
        # two layers
        depth = [0.02, 0.08]
        vel_s = [0.2, 0.6, 1.5]
    elif n_layers == 3:
        # three layers
        depth = [0.02, 0.04, 0.1]
        vel_s = [0.2, 0.6, 1.0, 1.5]

    model_params = setup_test_model(n_layers)
    data = setup_test_data(model_params, noise_dist, noise_params, depth, vel_s)

    inversion_init_kwargs = {
        "n_burn": 10000,
        "n_chunk": 500,
        "n_mcmc": 50000,
        "n_cov_chunk": 500,
        "n_thin": 10,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
        "set_starting_model": set_starting_model,
    }

    # for frequency dependent noise model, scale using observed data
    if inv_noise_params["frequency_scaling"]:
        if inv_noise_dist == "normal":
            inv_noise_params["std"] = inv_noise_params["std_percent"] * data.data_obs
        elif inv_noise_dist == "asym-laplace":
            inv_noise_params["lambd_scale"] = (
                inv_noise_params["lambd_scale_percent"] * data.data_obs
            )

    model_kwargs = {"noise_dist": inv_noise_dist, "noise_params": inv_noise_params}

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
    set_starting_model = True
    rotate = False

    n_layers = 1
    noise_dist = "normal"
    # noise_dist = "asym-laplace"
    inv_noise_dist = "normal"
    # inv_noise_dist = "asym-laplace"
    frequency_scaling = False

    noise_params = {"frequency_scaling": frequency_scaling}
    if noise_dist == "normal":
        std = 0.100  # 0.050 # 0.150 # km/s
        std_percent = 0.10

        if frequency_scaling:
            # for normal errors with frequency dependence,
            # use the percent of the data as the standard deviation
            noise_params["std_percent"] = std_percent
        else:
            # for IID errors, the value for normal standard deviation
            noise_params["std"] = std

    elif noise_dist == "asym-laplace":
        lambd_scale = 0.100  # 0.050 # 0.150 # km/s
        lambd_scale_percent = 0.10
        lambd, kappa = 5.6, 0.72

        noise_params["lambd"] = lambd
        noise_params["kappa"] = kappa
        if frequency_scaling:
            noise_params["lambd_scale_percent"] = lambd_scale_percent
        else:
            noise_params["lambd_scale"] = lambd_scale

    inv_noise_params = noise_params.copy()

    # currently set up to use same noise params for real noise and for model noise
    inversion, model_params = basic_inversion(
        n_layers=n_layers,
        noise_dist=noise_dist,
        noise_params=noise_params,
        inv_noise_dist=inv_noise_dist,
        inv_noise_params=inv_noise_params,
        set_starting_model=set_starting_model,
    )
    inversion.random_walk(
        model_params,
        proposal_distribution="cauchy",
        sample_prior=sample_prior,
        rotate_params=rotate,
    )


def plot_inversion(file_name):
    input_path = "./results/inversion/input-" + file_name + ".nc"
    results_path = "./results/inversion/results-" + file_name + ".nc"

    input_ds = xr.open_dataset(input_path)
    results_ds = xr.open_dataset(results_path)

    plot_results(input_ds, results_ds, out_filename=file_name, plot_true_model=True)


if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """

    run_inversion()

    # IID normal dist
    # file_name = "1778183486"

    # IID AL dist
    # file_name = "1778184831"

    # plot_inversion(file_name)
