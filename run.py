import numpy as np
from inversion import Inversion
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from velocity_model import VelocityModel

# TODO:
# - add environment
# - readme
# - figures folder
# - add tests
# - fix linters


def setup_scene(
    n_layers,
    poisson_ratio=0.265,
    density_params=[-1.91018882e-03, 1.46683536e04],
    sigma_pd=0.1,
    pcsd=0.05,
):
    """
    Define variables for initialization.
    """

    # Bounds of search (min, max)
    # density bounds
    bounds = {
        "layer_thickness": [0, 10],
        "v_p": [3, 7],
        "v_s": [2, 6],
        "sigma_pd": [0, 1],
    }
    # Initial velocity model

    true_model = VelocityModel.generate_true_model(
        n_layers, bounds["layer_thickness"], poisson_ratio, density_params, sigma_pd
    )

    # define freqs from layer thickness
    period = np.linspace(1, 100, 10)
    freqs = 3 * 1 / period
    pd_rayleigh = VelocityModel.forward_model(freqs, true_model.velocity_model)
    # add noise
    avg_vs_obs = pd_rayleigh.velocity + sigma_pd * np.random.randn(n_layers)

    return true_model, avg_vs_obs, bounds


def run(n_layers=10):
    """
    Run inversion.
    """
    _, avg_vs_obs, bounds = setup_scene(n_layers)
    starting_model = VelocityModel.generate_starting_model(true_model, bounds, n_params)
    inversion = Inversion(
        avg_vs_obs,
        starting_model,
        bounds,
        n_layers,
        n_chains=2,
    )
    resulting_model = inversion.run_inversion()


def create_test_data():
    pass


def plot_results():
    # TODO:
    # add units
    # plot ellipticity
    # save figures directly to folder
    # plot density

    true_model, avg_vs_obs, bounds = setup_scene(n_layers=10)

    # periods = np.linspace(1, 100, 100)  # unit
    # freqs = 1 / periods

    # pd_rayleigh = prior_model.get_rayleigh_phase_dispersion(periods)

    plot_model_setup(true_model, avg_vs_obs)
    # plot density
    # plot true model with and without noise

    # plot_velocity_profile(prior_model, periods)

    # plot_model_setup(prior_model.velocity_model)

    # plot_depth(periods, pd_rayleigh.velocity)

    # make simulated data, save to csv


def plot_model_setup(velocity_model, avg_vs_obs):
    # plot velocity_model
    # thickness, Vp, Vs, density
    # km, km/s, km/s, g/cm3

    depth = np.cumsum(velocity_model.thickness)

    plt.subplot(2, 2, 1)
    plt.scatter(velocity_model.vel_p, depth)
    # plt.axhline(thickness)
    plt.gca().invert_yaxis()
    plt.xlabel("P wave velocity (km/s)")
    plt.ylabel("depth (km)")

    plt.subplot(2, 2, 2)
    plt.scatter(velocity_model.vel_s, depth)
    plt.gca().invert_yaxis()
    plt.xlabel("S wave velocity (km/s)")
    plt.ylabel("depth (km)")

    plt.subplot(2, 2, 3)
    plt.scatter((velocity_model.density), depth)
    plt.ticklabel_format(style="sci", scilimits=(-2, 2))
    plt.gca().invert_yaxis()
    plt.xlabel("density (g/cm3)")
    plt.ylabel("depth (km)")

    plt.subplot(2, 2, 4)
    plt.scatter(avg_vs_obs, depth)
    plt.gca().invert_yaxis()
    plt.xlabel("avg vs observed (km/s)")
    plt.ylabel("depth (km)")

    plt.tight_layout()
    plt.show()


def plot_phase_dispersion():
    pass


def plot_velocity_profile(prior_model, periods):

    depths, vel_s_profile = prior_model.get_vel_s_profile(periods)

    plt.scatter(vel_s_profile, depths)
    plt.gca().invert_yaxis()
    plt.xlabel("shear velocity (km/s)")
    plt.ylabel("depth (km)")
    plt.tight_layout()
    plt.show()


def plot_depth(periods, vel):
    freq = 1 / periods

    wavelengths = vel / freq

    plt.subplot(2, 1, 1)
    plt.scatter(freq, wavelengths)
    plt.xlabel("frequency")
    plt.ylabel("wavelength")

    plt.subplot(2, 1, 2)
    plt.scatter(periods, wavelengths)
    plt.xlabel("period")
    plt.ylabel("wavelength")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results()
