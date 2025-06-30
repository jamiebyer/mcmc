import cProfile
from pstats import Stats, SortKey
import os
import numpy as np

from inversion.data import SyntheticData
from inversion.model_params import DispersionCurveParams
from inversion.inversion import Inversion

from plotting.plot_inversion import *


def setup_data(model_params, sigma_data):
    n_data = 50
    periods = np.flip(1 / np.logspace(0, 1.1, n_data))

    # run synthetic data that uses inversion calculations for vel_p and density
    # and optionally, setting vel_p and density exactly.
    thickness = [0.03, 0.05]
    vel_s = [0.4, 1.5, 2.0]

    # thickness = [0.03]
    # vel_s = [0.4, 1.5]

    data = SyntheticData(
        periods,
        sigma_data,
        model_params,
        thickness=thickness,
        vel_s=vel_s,
    )

    return data


def run_inversion(
    noise,
    model_params_kwargs,
    model_kwargs,
    inversion_init_kwargs,
    inversion_run_kwargs,
):
    """
    run tests for:
    - sample prior; uniform
    - sample prior; cauchy
    - start from true; low noise
    - start from true; high noise
    - generate true model; low noise
    - generate true model; high noise
    """

    # model params
    model_params = DispersionCurveParams(**model_params_kwargs)
    data = setup_data(model_params, sigma_data=noise)

    # run inversion
    inversion = Inversion(
        data,
        model_params,
        **model_kwargs,
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
        out_filename=file_name,
        rotation=False,
        sample_prior=sample_prior,
    )


if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """

    sample_prior = False
    proposal_distribution = "cauchy"
    set_starting_model = False

    noise = 0.1
    sigma_data = 0.1
    posterior_width = {
        "thickness": 0.05,
        "vel_s": 0.05,
    }  # fractional step size (multiplied by param bounds width)

    file_name = "layers-2"
    out_path = "./results/inversion/results-" + file_name + ".nc"

    rerun, plot = True, True

    if rerun:
        # set up data and inversion params
        bounds = {
            "thickness": [0.001, 0.1],  # km
            "vel_s": [0.1, 1.8],  # km/s
        }
        model_params_kwargs = {
            "n_layers": 2,
            "vpvs_ratio": 1.75,
            "param_bounds": bounds,
            "posterior_width": posterior_width,
        }
        model_kwargs = {"sigma_data": sigma_data}
        inversion_init_kwargs = {
            "n_burn": 0,
            "n_chunk": 500,
            "n_mcmc": 50000,
            "n_chains": 1,
            "beta_spacing_factor": 1.15,
        }
        inversion_run_kwargs = {
            "proposal_distribution": proposal_distribution,
        }

        run_inversion(
            noise,
            model_params_kwargs,
            model_kwargs,
            inversion_init_kwargs,
            inversion_run_kwargs,
        )
    if plot:
        # plot_inversion_results_param_prob(out_path, skip_inds=4000)
        # plot_inversion_results_param_time(out_path, skip_inds=4000)
        # plot_pred_vs_obs(out_path, skip_inds=4000)
        # plot_pred_hist(out_path, skip_inds=4000)
        plot_resulting_model_hist(out_path, skip_inds=4000)
