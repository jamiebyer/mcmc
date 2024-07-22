import numpy as np
from inversion import Inversion
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from model import Model, TrueModel, ChainModel

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
    n_chains,
    n_layers,
    n_data,
    sigma_pd,
    poisson_ratio=0.265,
    density_params=[-1.91018882e-03, 1.46683536e04],
):
    """
    Define parameter bounds, frequencies. Create simulated true model and observed data.

    :param n_layers: Number of layers in model.
    :param n_data: Number of simulated observed data.
    :param sigma_pd: Initial estimate for uncertainty in phase velocity.
    :param poisson_ratio:
    :density_params: Birch parameters to use for initial density profile.
    """
    # *** might move n_bins, n_keep to inversion class. ***

    # Bounds of search (min, max)
    # *** add units ***
    layer_bounds = [5, 15]  # units
    vel_s_bounds = [2, 6]
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
    freqs = np.linspace(1600, 400, n_data)  # (Hz)

    true_model = TrueModel(
        n_layers,
        n_data,
        freqs,
        sigma_pd,
        layer_bounds,
        poisson_ratio,
        density_params,
    )

    return freqs, poisson_ratio, density_params, layer_bounds, param_bounds, true_model


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
    freqs, poisson_ratio, density_params, layer_bounds, param_bounds, true_model = (
        setup_scene(
            n_chains,
            n_layers,
            n_data,
            model_depth,
            sigma_pd,
        )
    )

    # run inversion
    inversion = Inversion(
        n_chains,
        n_data,
        n_layers,
        layer_bounds,
        param_bounds,
        poisson_ratio,
        density_params,
        sigma_pd,
        freqs,
        true_model.phase_vel_obs,
        n_bins,
        n_burn,
        n_keep,
        n_rot,
    )
    # inversion.random_walk()

    # plots and comparing results to true model
    plot_results(true_model)


def plot_results(true_model):
    # plot true s and p velocitys, plot observed phase velocity, plot density profile.
    plot_scene(true_model)

    # plot convergence...
    # plot PT swapping
    # plot likelihood against steps
    # plot model parameters over time
    # plot residuals of phase_vel_obs against steps
    # acceptance rate
    # step size...


def plot_scene(true_model):
    depths = np.cumsum(true_model.thickness)

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    # plot true model velocities
    ax1.plot(true_model.vel_s, depths, label="s velocity")
    ax1.plot(true_model.vel_p, depths, label="p velocity")
    ax1.invert_yaxis()
    ax1.set_xlabel("km/s")
    ax1.set_ylabel("depth (km)")
    ax1.set_title("sigma_pd=" + str(true_model.sigma_pd))
    ax1.legend(loc="best")
    [ax1.axhline(y=d, c="black", alpha=0.25) for d in depths]

    ax2.plot(true_model.density, depths, label="density")
    ax2.set_xlabel("g/cm^3")
    ax2.legend(loc="best")
    [ax2.axhline(y=d, c="black", alpha=0.25) for d in depths]

    # plot phase velocity against frequency
    # ax1.plot(true_model.phase_vel_obs, depths, label="phase velocity")

    plt.tight_layout()
    plt.show()
    # save to file


def plot_histograms(inversion):

    for chain in inversion.chains:
        pass


if __name__ == "__main__":
    run(
        # n_chains=2,
        # n_data=10,
        # n_layers=10,
        # model_depth=20,
        # sigma_pd=0.0001,
        # n_bins=200,
        # n_burn=10000,
        # n_keep=2000,
        # n_rot=40000,
    )
