import numpy as np


class Event:
    def __init__(self, x, y, z, t_origin):
        self.x = x
        self.y = y
        self.z = z
        self.t_origin = t_origin
        self.t_obs_p = None
        self.t_obs_s = None

    def set_t_obs(self, n_stations, station_positions, prior_model):
        times_p, times_s = self.get_times(station_positions, prior_model)

        self.t_obs_p = times_p + prior_model.sigma_p * np.random.randn(n_stations)
        self.t_obs_s = times_s + prior_model.sigma_s * np.random.randn(n_stations)

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
