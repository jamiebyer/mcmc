import numpy as np
import forward_model


class Event:
    def __init__(self, x, y, z, t_origin):
        self.x = x
        self.y = y
        self.z = z
        self.t_origin = t_origin
        self.t_obs_p = None
        self.t_obs_s = None

    def get_t_obs(self, n_stations, station_positions, prior_model):
        times_p, times_s = self.get_times(station_positions, prior_model)

        self.t_obs_p = times_p + prior_model.sigma_p * np.random.randn(n_stations)
        self.t_obs_p = times_s + prior_model.sigma_s * np.random.randn(n_stations)

    def get_times(self, station_positions, prior_model):
        # for both S and P waves
        times_p = self.travel_time_3D(
            x=self.x,
            y=self.y,
            z=self.z,
            x_pos=station_positions["x"],
            y_pos=station_positions["y"],
            z_pos=station_positions["z"],
            vel=prior_model.vel_p,
            t_origin=self.t_origin,
        )
        times_s = self.travel_time_3D(
            x=self.x,
            y=self.y,
            z=self.z,
            x_pos=station_positions["x"],
            y_pos=station_positions["y"],
            z_pos=station_positions["z"],
            vel=prior_model.vel_s,
            t_origin=self.t_origin,
        )
        return times_p, times_s

    def travel_time_3D(x, y, z, x_pos, y_pos, z_pos, vel, t_origin):
        """
        x_pos: Station x locations.

        """
        t = (
            t_origin
            + np.sqrt((x_pos - x) ** 2 + (y_pos - y) ** 2 + (z_pos - z) ** 2) / vel
        )
        return t


def setup_model(n_stations=10):
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

    # Initial velocity model
    prior_model = forward_model.VelocityModel(
        vel_p=6.0, vel_s=3.47, sigma_p=0.05, sigma_s=0.3, thicknesses=[]
    )

    # Set t_obs for each event
    for event in events:
        event.get_t_obs(n_stations, station_positions, prior_model)

    # Calculate vel_s from prior model
    vel_s = prior_model.calculate_vel_s()
