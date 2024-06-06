import numpy as np
from forward_model import ForwardModel
from inversion import Inversion
from event import Event
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate


def get_initial_density(thickness):
    """
    Get initial densities from PREM csv.

    :param thickness: thickness of each layer, descending. (km)

    :return: densities for each layer. (g/cm3)
    """
    prem = pd.read_csv("./PREM500_IDV.csv")
    density = prem["density"]  # kg/m^3
    radius = prem["radius"] / 1000  # convert m to km

    depth = np.cumsum(thickness)

    x = np.arange(0, 10)
    y = np.exp(-x / 3.0)
    density_func = interpolate.interp1d(radius, density)
    density = density_func(depth)

    # VP = a + rho b # Birch

    return density  # g/cm3


def setup_scene(n_stations=10):
    """
    Define bounds, station positions...
    """

    # Bounds of search (min, max)
    bounds = {
        "lat": [0, 94],
        "lon": [0, 80],
        "depth": [0, 35],
        "t_origin": [0, 10],
        "v_p": [3, 7],
        "v_s": [2, 6],
        "sigma_p": [0, 1],
        "sigma_s": [0, 1],
    }

    # Station locations
    station_positions = {
        "lat": np.random.rand(n_stations) * (bounds["lat"][1] - bounds["lat"][0]),
        "lon": np.random.rand(n_stations) * (bounds["lon"][1] - bounds["lon"][0]),
        "depth": np.zeros(n_stations),
    }
    # Events
    events = [
        Event(lat=60, lon=35, depth=10, t_origin=1),
        Event(lat=70, lon=40, depth=15, t_origin=2),
    ]

    velocity_model = {
        "thickness": [10, 10, 10, 10, 10, 10, 10, 10, 10],  # km
        # "density": [2, 2, 2, 2, 2, 2, 2, 2, 2], # g/cm3
        "vel_p": [7.0, 6.8, 7.0, 7.6, 8.4, 9.0, 9.4, 9.6, 9.5],  # km/s
        "vel_s": [3.5, 3.4, 3.5, 3.8, 4.2, 4.5, 4.7, 4.8, 4.75],  # km/s
        "sigma_p": 0.05,
        "sigma_s": 0.3,
    }

    velocity_model["density"] = get_initial_density(thickness)

    # Initial velocity model
    prior_model = ForwardModel(**velocity_model)

    # Set t_obs for each event
    for event in events:
        event.set_t_obs(n_stations, station_positions, prior_model)

    return station_positions, events, prior_model, bounds


def run():
    """
    Run inversion.
    """
    station_positions, events, prior_model, bounds = setup_scene()
    inversion = Inversion(
        station_positions,
        events,
        prior_model,
        bounds,
        n_chains=2,
        n_burn=10000,
        n_keep=2000,
        n_rot=40000,
    )
    resulting_model = inversion.run_inversion()


def plot_results():
    # TODO:
    # add units
    # plot ellipticity
    # save figures directly to folder

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
