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
    n_data,
    poisson_ratio=0.265,
    density_params=[540.6, 360.1],
    n_layers=10,
    sigma_pd_true=0.0001,
    layer_bounds=[5e-3, 15e-3],
    vel_s_bounds=[2, 6],
    sigma_pd_bounds=[0, 1],
    freq_range=[2.5, 1.5]
):
    """
    Define parameter bounds. Create simulated true model and observed data.

    :param poisson_ratio:
    :param density_params:
    :param n_layers: Number of layers in model.
    :param n_data: Number of simulated observed data.
    :param sigma_pd_true:

    # Bounds of search (min, max)
    :param layer_bounds: (km)
    :param vel_s_bounds: (km/s)
    :param sigma_pd_bounds: 
    :param freq_range: range of frequencies for simulated data. exponent values used to make a range of frequencies. (Hz)
    """

    # *** should there be bounds on density and vel_p even though they are calculated from vel_s, thickness ***
    # density_bounds = [2, 6]
    # vel_p_bounds = [3, 7],

    # reshape bounds to be the same shape as params
    param_bounds = np.concatenate(
        ([layer_bounds] * n_layers, [vel_s_bounds] * n_layers, [sigma_pd_bounds]),
        axis=0,
    )

    # add the range of the bounds to param_bounds as a third column (min, max, range)
    range = param_bounds[:, 1] - param_bounds[:, 0]
    param_bounds = np.column_stack((param_bounds, range))

    # frequencies for simulated observed data
    # *** how would this data be collected? what is the spacing between frequencies? ***
    # *** frequencies will be input from the data. total depth of the model should be similar to data depth ***
    freqs = np.logspace(freq_range[0], freq_range[1], n_data)  # (Hz)
    
    true_model = TrueModel(
        n_layers,
        n_data,
        freqs,
        param_bounds,
        poisson_ratio,
        density_params,
    )

    return param_bounds, true_model


def run(
    true_model_kwargs,
    inversion_kwargs,
    n_data=10,
    hist_conv=0.05,
    out_dir="/out/inversion_results",
):
    """
    Run inversion.

    :param true_model_kwargs: (poisson_ratio, density_params, n_layers, sigma_pd_true, layer_bounds, vel_s_bounds, sigma_pd_bounds)
        changeable variables that affect the true model (simulated data)
    :param inversion_kwargs: (poisson_ratio, density_params, n_chains, n_layers, n_bins, n_burn, n_keep, n_rot)
        changeable variables that affect the inversion.
    :param out_dir: directory where to save results
    """

    # generate the true model to be used in the inversion. this is setting values
    # that we would get from collecting data
    # this will be different for different types of inversion.
    param_bounds, true_model = setup_scene(n_data, true_model_kwargs)

    # plot_results(true_model)
    # run inversion
    inversion = Inversion(
        n_data, param_bounds, true_model.freqs, true_model.phase_vel_obs, inversion_kwargs
    )

    # *** should any of those params just be in random walk?
    asyncio.get_event_loop().run_until_complete(
        inversion.random_walk(hist_conv, out_dir)
    )


if __name__ == "__main__":
    """
    run from terminal
    """
    # n_layers is different between true model and inversion we won't know the number of layers in the model
    # poisson_ratio and density_params on both because generation for true model
    # and starting model should happen independently.
    true_model_kwargs = (
        {
            "poisson_ratio": 0.265,
            "density_params": [540.6, 360.1],  # *** check units ***
            "n_data": 10,
            "n_layers": 10,
            "layer_bounds": [5e-3, 15e-3]  # km
            "vel_s_bounds": [2, 6]  # km/s
            "sigma_pd_bounds": [0, 1]
        },
    )
    inversion_kwargs = (
        {
            "poisson_ratio": 0.265,
            "density_params": [540.6, 360.1],  # *** check units ***
            "n_chains": 2,
            "n_layers": 10,
            "n_bins": 200,
            "n_burn": 10000,
            "n_keep": 2000,
            "n_rot": 40000,
        },
    )

    run(
        true_model_kwargs,
        inversion_kwargs,
        n_data=10,
        hist_conv=0.05,
        out_dir="./out/inversion_results_" + str(time.time())
    )
