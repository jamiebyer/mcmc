import numpy as np
from inversion import Inversion
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from model import Model, TrueModel, ChainModel
import time
import asyncio

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
    :density_params: Birch parameters to use for initial density profile.
    # https://sci-hub.ru/10.1029/95jb00259
    # Seismic velocity structure and composition of the continental crust: A global view
    # Nikolas I. Christensen, Walter D. Mooney
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

    # plots and comparing results to true model
    # plot_results(true_model)


def plot_results(true_model):
    # plot true s and p velocities, plot observed phase velocity, plot density profile.
    plot_scene(true_model)

    # plot convergence...
    # plot PT swapping
    # plot likelihood against steps
    # plot model parameters over time
    # plot residuals of phase_vel_obs against steps
    # acceptance rate
    # step size...


def plot_scene(true_model):
    # *** get rid of repetition in plotting ***
    depths = np.cumsum(true_model.thickness) * 1000  # in m

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(12, 6))
    # plot true data
    data_phase_vel = true_model.phase_vel_true
    # data_depths = data_phase_vel / true_model.freqs  # *** check units ***
    # *** use the actual velocity ***
    data_depths = 3.5 / true_model.freqs  # *** check units ***
    ax1.plot(data_phase_vel, data_depths * 1000, label="true phase velocity")

    # plot observed data
    data_phase_vel = true_model.phase_vel_obs
    # data_depths = data_phase_vel / true_model.freqs  # *** check units ***
    data_depths = 3.5 / true_model.freqs  # *** check units ***
    # ax1.plot(data_phase_vel, data_depths * 1000, label="observed phase velocity")

    ax1.invert_yaxis()
    ax1.set_xlabel("km/s")
    ax1.ticklabel_format(style="scientific", scilimits=(-3, 3))
    ax1.set_ylabel("depth (m)")
    ax1.set_title("sigma_pd=" + str(true_model.sigma_pd))
    ax1.legend(loc="best")
    [ax1.axhline(y=d, c="black", alpha=0.25) for d in depths]

    # plot true model velocities
    ax2.plot(true_model.vel_s, depths, label="s velocity")
    ax2.plot(true_model.vel_p, depths, label="p velocity")

    ax2.ticklabel_format(style="scientific", scilimits=(-3, 3))
    ax2.set_xlabel("km/s")
    ax2.legend(loc="best")
    [ax2.axhline(y=d, c="black", alpha=0.25) for d in depths]

    # plot density depth profile
    ax3.plot(true_model.density, depths, label="density")
    ax3.ticklabel_format(style="scientific", scilimits=(-3, 3))
    ax3.set_xlabel("g/cm^3")
    ax3.legend(loc="best")
    [ax3.axhline(y=d, c="black", alpha=0.25) for d in depths]

    fig.tight_layout()
    # plt.show()
    # save to file
    # fig.savefig("./figures/scene.png")


def plot_histograms(inversion):
    # reading back from the file....
    pass


def plot_inversion_results(out_dir):
    inversion_results = pd.read_csv(out_dir + ".csv")

    # plot params
    # params_results = inversion_results["params"]
    # plt.plot(params_results)

    # plot logL
    logL_results = inversion_results["params"]
    plt.plot(logL_results)
    plt.show()
    # save to file
    # fig.savefig("./figures/scene.png")


if __name__ == "__main__":
    out_dir = "./out/inversion_results_" + str(time.time())
    # """
    run(
        # n_chains=2,
        n_data=15,
        # n_layers=10,
        sigma_pd_true=0.001,
        # n_bins=200,
        n_burn=1000,
        n_keep=200,
        n_rot=4000,
        out_dir=out_dir,
    )

    """

    plot_inversion_results(
        out_dir=r"/home/jbyer/Documents/school/grad/research/hvsr/hvsr/out/inversion_results_1722542488.3892229"
    )
    """
