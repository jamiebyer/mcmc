import numpy as np
from inversion import Inversion
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from velocity_model import Model, TrueModel, ChainModel

# TODO:
# - add environment
# - readme
# - figures folder
# - add tests
# - fix linters
# - docstrings and references to papers
# - alphabetize or organize function order
# - check variable names, rename u, pcsd
# - cauchy proposal...
# - maybe add getters for like velocity model
# - distinguish n_params, n_data
# - add logging
# - optimization


@staticmethod
def get_betas(n_temps, dTlog=1.15):
    """
    setting values for beta to be used. the first quarter of the chains have beta=0

    :param dTlog:

    :return beta: inverse temperature; beta values to use for each chain
    """
    # *** maybe this function should be replaced with a property later***
    # Parallel tempering schedule
    # the first ~1/4 of chains have beta value of 0..?
    n_temps_frac = int(np.ceil(n_temps / 4))
    betas = np.zeros(n_temps, dtype=float)

    inds = np.arange(n_temps_frac, n_temps)
    betas[inds] = 1.0 / dTlog**inds
    # T = 1.0 / beta

    return betas


def setup_scene(
    n_layers,
    n_data,
    n_chains,
    model_depth,
    sigma_pd,
    poisson_ratio=0.265,
    density_params=[-1.91018882e-03, 1.46683536e04],
):
    """
    Define parameter bounds, frequencies. Create simulated true model and observed data.

    :param n_layers: Number of layers in model.
    :param n_data: Number of simulated observed data.
    :param model_depth: Total depth of model. (m)
    :param sigma_pd: Initial estimate for uncertainty in phase velocity.
    :param poisson_ratio:
    :density_params: Birch parameters to use for initial density profile.
    """
    # *** might move n_bins, n_keep to inversion class. ***

    # *** add units ***
    # *** are we checking bounds for the other parameters? ***
    # Bounds of search (min, max)
    bounds_dict = {
        "layer_thickness": [5, 15],
        "vel_p": [3, 7],
        "vel_s": [2, 6],
        "density": [2, 6],
        "sigma_pd": [0, 1],
    }
    # bounds needs to be the same shape as params
    layer_bounds = np.repeat(bounds_dict["layer_thickness"], n_layers)
    vel_s_bounds = np.repeat(bounds_dict["vel_s"], n_layers)
    param_bounds = np.concatenate(
        (layer_bounds, vel_s_bounds, bounds_dict["sigma_pd"]), axis=1
    )

    # *** how would this data be collected? what is the spacing between frequencies? ***
    # frequencies for simulated observed data
    freqs = np.linspace(400, 1600, n_data)  # (Hz)

    # generate true model
    true_model_params = Model.generate_model_params(
        n_data, layer_bounds, poisson_ratio, density_params
    )
    true_model = TrueModel(freqs, n_data, *true_model_params, sigma_pd)

    # generate the starting models
    betas = get_betas(n_chains)
    starting_models = []
    for ind in range(n_chains):
        # *** generate params within bounds, poisson_ratio, and density_params ? ***
        # *** starting_model params should be separate from true model params ***
        starting_model_params = Model.generate_model_params(
            n_data, layer_bounds, poisson_ratio, density_params
        )
        model = ChainModel(betas[ind], *starting_model_params)
        starting_models.append(model)

    return (freqs, param_bounds, betas, true_model, starting_models)


def run(
    n_chains=2,
    n_data=10,
    n_layers=10,
    model_depth=20,
    sigma_pd=0.0001,
    n_bins=200,
    n_burn=10000,
    n_keep=2000,
    n_rot=40000,
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
    freqs, param_bounds, betas, true_model, starting_models = setup_scene(
        n_layers, n_data, model_depth, sigma_pd
    )

    # run inversion
    inversion = Inversion(
        freqs,
        n_layers,
        param_bounds,
        starting_models,
        true_model.phase_vel_obs,
        sigma_pd,
        # n_bins,
        # n_burn,
        # n_keep,
        # n_rot,
    )
    inversion.random_walk()

    # plots and comparing results to true model


if __name__ == "__main__":
    run(
        # n_chains=2,
        # n_data=10,
        # n_layers=10,
        # model_depth=20,
        # sigma_pd=0.0001,
    )
