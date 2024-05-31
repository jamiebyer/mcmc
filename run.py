import numpy as np
from forward_model import ForwardModel
from event import Event


def setup_forward_model(n_stations=10):
    # Bounds of search (min, max)
    x_bounds = [0, 94]
    y_bounds = [0, 80]

    # Station locations
    station_positions = {
        "x": np.random.rand(n_stations) * (x_bounds[1] - x_bounds[0]),
        "y": np.random.rand(n_stations) * (y_bounds[1] - y_bounds[0]),
        "z": np.zeros(n_stations),
    }
    # Events
    events = [
        Event(lat=60.0, lon=35.0, depth=10.0, t_origin=1.0),
        Event(lat=70.0, lon=40.0, depth=15.0, t_origin=2.0),
    ]

    # use dispersion curve to set priors for the model
    # vel_s = prior_model.calculate_vel_s()

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
    forward_model = ForwardModel(
        vel_p=6.0,
        vel_s=3.47,
        sigma_p=0.05,
        sigma_s=0.3,
        velocity_model=velocity_model,
    )

    # Set t_obs for each event
    for event in events:
        event.get_t_obs(n_stations, station_positions, forward_model)

    return forward_model


def run():
    pass


def plot_results():
    forward_model = setup_forward_model()

    # plot vs. freq, wavelength, depth
    # plot of velocity model
    # plot ellipticity
    pass
