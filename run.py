import numpy as np
from inversion import Inversion
from event import Event
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from velocity_model import generate_true_model, generate_starting_model

# TODO:
# - add environment
# - readme
# - figures folder
# - add tests


def setup_scene(n_layers=10, poisson_ratio=0.265, density_params=None, pcsd=1 / 20):
    """
    Define bounds, station positions...
    """

    # Bounds of search (min, max)
    # density bounds
    bounds = {
        "layer_thickness": [0, 10],
        "v_p": [3, 7],
        "v_s": [2, 6],
        "sigma_p": [0, 1],
        "sigma_s": [0, 1],
    }
    # Initial velocity model
    true_model = generate_true_model(
        n_layers,
        bounds["layer_thickness"],
        poisson_ratio,
    )

    pd_rayleigh = true_model.forward_model()
    avg_vs_obs = pd_rayleigh.velocity + sigma_pd * np.random.randn(n_layers)

    starting_model = generate_starting_model(true_model, pcsd, n_layers)

    return avg_vs_obs, starting_model, bounds, n_layers


def run():
    """
    Run inversion.
    """
    avg_vs_obs, starting_model, bounds, n_layers = setup_scene()
    inversion = Inversion(
        avg_vs_obs,
        starting_model,
        bounds,
        n_layers,
        n_chains=2,
    )
    resulting_model = inversion.run_inversion()


def plot_results():
    # TODO:
    # add units
    # plot ellipticity
    # save figures directly to folder
    # plot density

    station_positions, events, prior_model, bounds = setup_scene()
    periods = np.linspace(1, 100, 100)  # unit
    pd_rayleigh = prior_model.get_rayleigh_phase_dispersion(periods)

    plot_model_setup(prior_model.velocity_model, station_positions)

    # plot_velocity_profile(prior_model, periods)

    # plot_model_setup(prior_model.velocity_model)

    # plot_depth(periods, pd_rayleigh.velocity)

    # make simulated data, save to csv


def plot_model_setup(velocity_model, station_positions):
    # plot velocity_model
    # thickness, Vp, Vs, density
    # km, km/s, km/s, g/cm3

    depth = np.cumsum(velocity_model[:, 0])
    vel_p = velocity_model[:, 1]
    vel_s = velocity_model[:, 2]
    density = velocity_model[:, 3]

    plt.subplot(1, 2, 1)
    plt.scatter(vel_p, depth, label="vp")
    plt.scatter(vel_s, depth, label="vs")
    plt.gca().invert_yaxis()
    plt.xlabel("velocity (km/s)")
    plt.ylabel("depth (km)")

    plt.subplot(1, 2, 2)
    plt.scatter(density, depth)
    plt.gca().invert_yaxis()
    plt.xlabel("density (g/cm3)")
    plt.ylabel("depth (km)")

    # plot station locations
    plt.tight_layout()
    plt.show()

    plt.subplot(1, 3, 1)
    plt.scatter(station_positions["lat"], station_positions["lon"])
    plt.xlabel("lat")
    plt.xlabel("lon")

    plt.subplot(1, 3, 2)
    plt.scatter(station_positions["lat"], station_positions["depth"])
    plt.xlabel("lat")
    plt.xlabel("depth")

    plt.subplot(1, 3, 3)
    plt.scatter(station_positions["lon"], station_positions["depth"])
    plt.xlabel("lon")
    plt.xlabel("depth")

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
