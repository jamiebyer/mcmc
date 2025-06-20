import pytest
import numpy as np

from plotting.plot_inversion import (
    plot_inversion_results_param_prob,
    plot_inversion_results_param_time,
)
import matplotlib.pyplot as plt
import xarray as xr
import os
from inversion.data import SyntheticData
from inversion.model import Model
from inversion.inversion import Inversion
from inversion.model_params import DispersionCurveParams

import cProfile
from pstats import Stats, SortKey

np.random.seed(0)


def setup_data(sigma_data):
    n_data = 50
    periods = np.flip(1 / np.logspace(0, 1.1, n_data))
    data_kwargs = {
        "thickness": [0.03],
        "vel_s": [0.4, 1.5],
        "vel_p": [1.6, 2.5],
        "density": [2.0, 2.5],
    }

    data = SyntheticData(periods, sigma_data, **data_kwargs)
    # plt.plot(data.periods, data.data_true, "-b")
    # plt.plot(data.periods, data.data_obs, "x")
    # plt.show()

    return data


def setup_true_model():
    pass


# TESTING BAYESIAN INVERSION


def test_acceptance_criteria():
    # try proposing a model out of bounds
    # rejection records the same model twice
    pass


def test_sampling_prior(rerun=False, plot=True, profiling=True):
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
    set_starting_model = True
    noise = 0.1
    sigma_model = {
        "thickness": 0.1,
        "vel_s": 0.1,
    }  # fractional step size (multiplied by param bounds width)

    in_path = "./results/inversion/results-sample_prior.nc"
    # likelihood always returns 1
    if rerun:
        # deleting test file if it exists
        if os.path.isfile(in_path):
            # *** make sure .nc is written in a context manager by the end
            f = open(in_path, "r")
            f.close()
            os.remove(in_path)

        # set up data and inversion params
        data = setup_data(sigma_data=noise)
        bounds = {
            "thickness": [0.001, 0.1],  # km
            "vel_s": [0.1, 1.8],  # km/s
        }
        model_params_kwargs = {
            "n_layers": 1,
            "sigma_model": sigma_model,
            "vpvs_ratio": 1.75,
            "param_bounds": bounds,
        }
        inversion_init_kwargs = {
            "n_burn": 0,
            "n_chunk": 500,
            "n_mcmc": 10000,
            "n_chains": 1,
            "beta_spacing_factor": 1.15,
        }
        inversion_run_kwargs = {
            "proposal_distribution": proposal_distribution,
            "scale_factor": [1, 1],
        }

        # model params
        model_params = DispersionCurveParams(**model_params_kwargs)

        # run inversion
        inversion = Inversion(
            data,
            model_params,
            **inversion_init_kwargs,
        )

        if set_starting_model:
            # set initial model to true model
            model = inversion.chains[0]
            test_model_params = np.array([0.03, 0.4, 1.5])
            velocity_model = model.model_params.get_velocity_model(test_model_params)
            model.model_params.model_params = test_model_params

            # set initial likelihood
            model.logL, model.data_pred = model.get_likelihood(velocity_model, data)

        # run inversion but always accept
        inversion.random_walk(
            model_params,
            **inversion_run_kwargs,
            out_filename="sample_prior",
            rotation=False,
            sample_prior=sample_prior,
        )

    # assert, all model params should be in bounds
    if plot:
        plot_inversion_results_param_prob(in_path)  # , skip_inds=500000)
        plot_inversion_results_param_time(in_path)  # , skip_inds=500000)


def test_low_noise_data(rerun=True, plot=True):
    in_path = "./results/inversion/results-test_low_noise.nc"
    # likelihood always returns 1
    if rerun:
        # deleting test file if it exists
        if os.path.isfile(in_path):
            # *** make sure .nc is written in a context manager by the end
            f = open(in_path, "r")
            f.close()
            os.remove(in_path)

        # set up data and inversion params
        # data = setup_data(sigma_data=0.01)
        data = setup_data(sigma_data=0.1)
        bounds = {
            "thickness": [0.001, 0.1],  # km
            "vel_s": [0.1, 1.8],  # km/s
        }
        model_params_kwargs = {
            "n_layers": 1,
            # "sigma_model": {"thickness": 0.1, "vel_s": 0.01},  # km  # km/s
            "sigma_model": {"thickness": 1, "vel_s": 1},  # km  # km/s
            "vpvs_ratio": 1.75,
            "param_bounds": bounds,
        }
        inversion_init_kwargs = {
            "n_burn": 0,
            "n_chunk": 500,
            "n_mcmc": 10000,
            "n_chains": 1,
            "beta_spacing_factor": 1.15,
        }
        inversion_run_kwargs = {
            # "proposal_distribution": "uniform",
            "proposal_distribution": "cauchy",
            "scale_factor": [1, 1],
        }

        # model params
        model_params = DispersionCurveParams(**model_params_kwargs)

        # run inversion
        inversion = Inversion(
            data,
            model_params,
            **inversion_init_kwargs,
        )

        """
        # set initial model to true model
        model = inversion.chains[0]
        test_model_params = np.array([0.03, 0.4, 1.5])
        velocity_model = model.model_params.get_velocity_model(test_model_params)
        model.model_params.model_params = test_model_params

        # set initial likelihood
        model.logL, model.data_pred = model.get_likelihood(velocity_model, data)
        """
        # run inversion but always accept
        inversion.random_walk(
            model_params,
            **inversion_run_kwargs,
            out_filename="test_low_noise",
        )

    if plot:
        plot_inversion_results_param_prob(in_path)  # , skip_inds=500000)
        plot_inversion_results_param_time(in_path)  # , skip_inds=500000)


def test_high_noise_data(rerun=False, plot=False):
    """
    Start at the true model for high noise case.
    """
    # set the initial model to the true model

    # check that (some logL values are better than true model, most are worse)

    data = setup_data(sigma_data=0.01)

    bounds = {
        "thickness": [0.01, 1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 5],  # km/s
        "density": [0.5, 5],  # g/cm^3
    }
    model_kwargs = {
        "n_layers": 2,
        "poisson_ratio": 0.265,
        "sigma_model": 0.005,
        "beta": 1,
        "n_bins": 200,
    }

    model = Model(**model_kwargs)

    # set initial model to true model
    model.thickness = np.array([0.03])
    model.vel_s = np.array([0.4, 1.5])
    model.vel_p = np.array([1.6, 2.5])
    model.density = np.array([2.0, 2.5])

    velocity_model = np.array([[0.03, 0], [1.6, 2.5], [0.4, 1.5], [2.0, 2.5]])

    # set initial likelihood
    model.logL, model.data_pred = model.get_likelihood(
        data.periods, velocity_model, data.data_obs
    )

    # check acceptance rate. (ratio of accepted to proposed steps)
    # can track the average over the whole run and that of the last 1000 or 10,000 steps.
    # expect / want an acceptance rate of ~0.3
    n_steps = 1000
    for _ in range(n_steps):
        # if using the perturb params function, it does the acceptance rate stats in the function
        model.perturb_params(bounds, data.periods, data.data_obs)

    assert (model.swap_acc + model.swap_prop) == n_steps
    assert model.swap_acc > 0.1 * n_steps
    assert model.swap_prop > 0.1 * n_steps

    # plotting
    if plot:
        # Generate a plot of data fit. That should include best fit model and a number of other fits that are also part of the sample. This is useful to look at to see if predictions are plausible.
        pass


def test_distributions():
    # test uniform vs. cauchy?
    # test that sampling prior goes to approximate distribution?
    pass


# TESTING OPTIMIZATION INVERSION


def test_constant_temp_inversion(setup_data):
    data = setup_data
    bounds = {
        "thickness": [0.01, 1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 5],  # km/s
        "density": [0.5, 5],  # g/cm^3
    }
    model_kwargs = {
        "n_layers": 2,
        "poisson_ratio": 0.265,
        "sigma_model": {"thickness": 0.1, "vel_s": 0.01},  # km  # km/s
        "beta": 1,
        "n_bins": 200,
    }
    model = Model(**model_kwargs)

    model.thickness = np.random.uniform(
        bounds["thickness"][0], bounds["thickness"][1], 1
    )

    model.vel_s = np.random.uniform(bounds["vel_s"][0], bounds["vel_s"][1], 2)
    valid_params = False
    while not valid_params:
        velocity_model, valid_params = model.get_velocity_model(
            bounds, model.thickness, model.vel_s
        )

    # set initial likelihood
    model.logL, model.data_pred = model.get_likelihood(velocity_model, data)
    # run inversion optimization
    model.get_optimization_model(bounds, data)

    # check acceptance rate over time?
    # check that it converges


def test_optimization_inversion(setup_data):
    data = setup_data
    bounds = {
        "thickness": [0.01, 1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 5],  # km/s
        "density": [0.5, 5],  # g/cm^3
    }
    model_kwargs = {
        "n_layers": 2,
        "poisson_ratio": 0.265,
        "sigma_model": {"thickness": 0.1, "vel_s": 0.01},  # km  # km/s
        "beta": 1,
        "n_bins": 200,
    }
    model = Model(**model_kwargs)

    model.thickness = np.random.uniform(
        bounds["thickness"][0], bounds["thickness"][1], 1
    )

    model.vel_s = np.random.uniform(bounds["vel_s"][0], bounds["vel_s"][1], 2)
    valid_params = False
    while not valid_params:
        velocity_model, valid_params = model.get_velocity_model(
            bounds, model.thickness, model.vel_s
        )

    # set initial likelihood
    model.logL, model.data_pred = model.get_likelihood(velocity_model, data)
    # run inversion optimization
    model.get_optimization_model(bounds, data)

    # check acceptance rate over time?
    # check that it converges


def test_mcmc_inversion(setup_data):
    data = setup_data

    bounds = {
        "thickness": [0.01, 1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 5],  # km/s
        "density": [0.5, 5],  # g/cm^3
    }
    model_kwargs = {
        "n_layers": 2,
        "sigma_model": 0.005,
        "poisson_ratio": 0.265,
        "param_bounds": bounds,
    }
    inversion_init_kwargs = {
        "n_bins": 200,
        "n_burn": 10000,
        "n_keep": 100,
        "n_rot": 40000,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
    }
    inversion_run_kwargs = {
        "max_perturbations": 10,
        "hist_conv": 0.05,
    }

    # run inversion
    inversion = Inversion(
        data,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    inversion.random_walk(**inversion_run_kwargs)
