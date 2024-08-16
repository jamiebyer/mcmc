import numpy as np
from inversion import Inversion
from model import TrueModel
import time
import asyncio

"""
TODO:
- add environment
- readme
- figures folder
- add tests
- fix linters
- docstrings and references to papers
- alphabetize or organize function order
- cauchy proposal...
- add logging
- optimization
- clear terminal errors
"""


def setup_scene(
    n_layers,
    n_data,
    sigma_pd_true,
    poisson_ratio=0.265,
    density_params=[540.6, 360.1],  # *** check units ***
    # density_params=[989.3, 289.1],
):
    """
    Define parameter bounds, frequencies. Create simulated true model and observed data.

    :param n_layers: Number of layers in model.
    :param n_data: Number of simulated observed data.
    :param sigma_pd:
    :param poisson_ratio:
    """
    # *** might move n_bins, n_keep to inversion class. ***

    # Bounds of search (min, max)
    # *** add units ***
    layer_bounds = [5e-3, 15e-3]  # km
    vel_s_bounds = [2, 6]  # km/s
    sigma_pd_bounds = [0, 1]

    # *** can i still enforce these bounds? ***
    # density_bounds = [2, 6]
    # vel_p_bounds = [3, 7],

    # bounds needs to be the same shape as params
    param_bounds = np.concatenate(
        ([layer_bounds] * n_layers, [vel_s_bounds] * n_layers, [sigma_pd_bounds]),
        axis=0,
    )
    # add bounds range to param_bounds
    range = param_bounds[:, 1] - param_bounds[:, 0]
    # param_bounds = np.append(param_bounds, range, axis=1)
    param_bounds = np.column_stack((param_bounds, range))

    # *** how would this data be collected? what is the spacing between frequencies? ***
    # *** or should it be periods... ***
    # frequencies for simulated observed data

    # frequencies will be input from the data. total depth of the model should be similar to data depth ***
    # freqs = np.linspace(1600, 400, n_data)  # (Hz)
    # freqs = np.logspace(0.5, -1.15, n_data)  # (Hz)
    freqs = np.logspace(2.5, 1.5, n_data)  # (Hz)

    true_model = TrueModel(
        sigma_pd_true,
        n_layers,
        n_data,
        freqs,
        param_bounds,
        poisson_ratio,
        density_params,
    )

    return freqs, poisson_ratio, density_params, param_bounds, true_model


def run(
    n_chains=2,
    n_data=10,
    n_layers=10,
    sigma_pd_true=0.0001,
    n_bins=200,
    n_burn=10000,
    n_keep=2000,
    n_rot=40000,
    out_dir="/out/inversion_results",
):
    """
    Run inversion.

    :param n_chains: Number of parallel chains to use in parallel tempering.
    :param n_data: Number of simulated data to use.
    :param n_layers: Number of layers in the model.
    :param model_depth: Total depth of model. (m)
    :param sigma_pd: Starting uncertainty for phase velocity data.
    """

    # declare parameters needed for inversion; generate true model
    freqs, poisson_ratio, density_params, param_bounds, true_model = setup_scene(
        n_layers,
        n_data,
        sigma_pd_true,
    )

    # plot_results(true_model)
    # run inversion
    inversion = Inversion(
        n_chains,
        n_data,
        n_layers,
        param_bounds,
        poisson_ratio,
        density_params,
        freqs,
        true_model.phase_vel_obs,
        n_bins,
        n_burn,
        n_keep,
        n_rot,
    )
    # should any of those params just be in random walk?
    # *** hist_conv values ***

    asyncio.get_event_loop().run_until_complete(
        inversion.random_walk(hist_conv=0.05, out_dir=out_dir)
    )


if __name__ == "__main__":
    out_dir = "./out/inversion_results_" + str(time.time())
    run(
        # n_chains=2,
        n_data=15,
        # n_layers=10,
        sigma_pd_true=0.001,
        # n_bins=200,
        n_burn=800,
        n_keep=30,
        n_rot=4000,
        out_dir=out_dir,
    )
