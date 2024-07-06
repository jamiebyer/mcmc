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
# - docstrings and references to papers


def setup_scene(
    freqs,
    sigma_pd,
    poisson_ratio=0.265,
    density_params=[-1.91018882e-03, 1.46683536e04],
):
    """
    Define bounds, frequencies. Create true model and observed data.
    """

    # Bounds of search (min, max)
    # density bounds
    bounds = {
        "layer_thickness": [5, 15],
        "v_p": [3, 7],
        "v_s": [2, 6],
        "sigma_pd": [0, 1],
    }
    # Initial velocity model
    n_freqs = len(freqs)
    true_model = VelocityModel.generate_true_model(
        n_freqs, bounds["layer_thickness"], poisson_ratio, density_params, sigma_pd
    )

    pd_rayleigh = VelocityModel.forward_model(freqs, true_model.velocity_model)
    # add noise
    phase_vel_true = pd_rayleigh.velocity
    phase_vel_obs = phase_vel_true + sigma_pd * np.random.randn(n_freqs)

    return true_model, phase_vel_true, phase_vel_obs, bounds


def run(n_layers=10):
    """
    Run inversion.
    """
    n_chains = 2
    sigma_pd = 0.0001
    periods = np.linspace(1, 100, 100)  # unit
    freqs = 1 / periods
    true_model, _, phase_vel_obs, bounds = setup_scene(freqs, n_layers, sigma_pd)
    chains = VelocityModel.generate_starting_models(
        n_chains, freqs, true_model, phase_vel_obs, bounds, sigma_pd
    )
    inversion = Inversion(phase_vel_obs, chains, bounds, n_layers)
    inversion.run_inversion(freqs, bounds, sigma_pd)


def plot_results():
    # TODO:
    # add units
    # plot ellipticity
    # save figures directly to folder
    # plot density

    true_model, phase_vel_true, phase_vel_obs, bounds = setup_scene()

    # periods = np.linspace(1, 100, 100)  # unit
    # freqs = 1 / periods

    # pd_rayleigh = prior_model.get_rayleigh_phase_dispersion(periods)

    plot_model_setup(true_model, phase_vel_true, phase_vel_obs, bounds)


def plot_model_setup(velocity_model, phase_vel_true, phase_vel_obs, bounds):
    # plot velocity_model
    # thickness, Vp, Vs, density
    # km, km/s, km/s, g/cm3

    depth = np.cumsum(velocity_model.thickness)

    plt.subplot(2, 2, 1)
    plt.scatter(velocity_model.vel_p, depth)
    # plt.axvline(bounds[])
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
    plt.scatter(avg_vs_true, depth, label="true")
    plt.scatter(avg_vs_obs, depth, label="obs")
    for d in depth:
        plt.axhline(d)
    plt.gca().invert_yaxis()
    plt.xlabel("avg vs observed (km/s)")
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
    run()
    # plot_results()
