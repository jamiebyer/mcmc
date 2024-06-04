import numpy as np
from forward_model import ForwardModel
from inversion import Inversion
from event import Event


def setup_scene(n_stations=10):
    """
    t: array of periods
    """
    # Bounds of search (min, max)
    bounds = {
        "x": [0, 94],
        "y": [0, 80],
        "depth": [0, 35],
        "t_origin": [0, 10],
        "v_p": [3, 7],
        "v_s": [2, 6],
        "sigma_p": [0, 1],
        "sigma_s": [0, 1],
    }

    # get bounds range

    # Station locations
    station_positions = {
        "x": np.random.rand(n_stations) * (bounds["x"][1] - bounds["x"][0]),
        "y": np.random.rand(n_stations) * (bounds["y"][1] - bounds["y"][0]),
        "z": np.zeros(n_stations),
    }
    # Events
    events = [
        Event(lat=60.0, lon=35.0, depth=10.0, t_origin=1.0),
        Event(lat=70.0, lon=40.0, depth=15.0, t_origin=2.0),
    ]

    """
    velocity_model = np.array(
        [
            [10.0, 7.00, 3.50, 2.00],
            [10.0, 6.80, 3.40, 2.00],
            [10.0, 7.00, 3.50, 2.00],
            [10.0, 7.60, 3.80, 2.00],
            [10.0, 8.40, 4.20, 2.00],
            [10.0, 9.00, 4.50, 2.00],
            [10.0, 9.40, 4.70, 2.00],
            [10.0, 9.60, 4.80, 2.00],
            [10.0, 9.50, 4.75, 2.00],
        ]
    )
    """
    # Initial velocity model
    prior_model = ForwardModel(
        vel_p=6.0,
        vel_s=3.47,
        sigma_p=0.05,
        sigma_s=0.3,
        velocity_model=velocity_model,
    )

    # Set t_obs for each event
    for event in events:
        event.set_t_obs(n_stations, station_positions, prior_model)

    return station_positions, events, prior_model, bounds


def run():
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


def plot_results():
    station_positions, events, prior_model, bounds = setup_scene()

    # plot vs. freq, wavelength, depth
    # plot of velocity model
    # plot ellipticity
    periods = np.linspace(0, 100, 100)

    pd_rayleigh = prior_model.get_rayleigh_phase_dispersion(periods)

    vel_s_profile = prior_model.get_vel_s_profile(periods)

    # make simulated data, save to csv
