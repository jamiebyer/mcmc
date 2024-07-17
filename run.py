import numpy as np
from inversion import Inversion
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from velocity_model import TrueModel, ChainModel

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


def setup_scene(
    n_layers,
    n_data,
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

    # *** add units ***
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
    true_model = TrueModel(
        n_data, layer_bounds, poisson_ratio, density_params, sigma_pd
    )

    return (
        freqs,
        bounds,
        true_model,
    )


def run(
    n_chains=2,
    n_data=10,
    n_layers=10,
    model_depth=20,
    sigma_pd=0.0001,
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
    freqs, bounds, true_model = setup_scene(n_layers, n_data, model_depth, sigma_pd)

    # create freqs, data matrix to give to inversion?
    phase_vel_obs = true_model.phase_vel_obs

    # *** starting models should be generate within inversion. currently they are generated from the true model... how should they be generated? ***
    # *** move generate_starting models to inversion init (and get n_keep from inversion) ***
    chains = ChainModel.generate_starting_models(
        n_chains, freqs, true_model, param_bounds, sigma_pd, n_keep=200
    )

    # run inversion
    inversion = Inversion(chains, bounds, n_layers)
    inversion.run_inversion(freqs, phase_vel_obs, bounds, sigma_pd)


if __name__ == "__main__":
    run(
        # n_chains=2,
        # n_data=10,
        # n_layers=10,
        # model_depth=20,
        # sigma_pd=0.0001,
    )
