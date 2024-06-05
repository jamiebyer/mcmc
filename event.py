import numpy as np


class Event:
    def __init__(self, lat, lon, depth, t_origin):
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.t_origin = t_origin
        self.t_obs_p = None
        self.t_obs_s = None

    def set_t_obs(self, n_stations, station_positions, prior_model):
        times_p, times_s = self.get_times(station_positions, prior_model)

        self.t_obs_p = times_p + prior_model.sigma_p * np.random.randn(n_stations)
        self.t_obs_s = times_s + prior_model.sigma_s * np.random.randn(n_stations)

    def get_times(self, station_positions, forward_model):
        # for both S and P waves
        times_p = Event.travel_time_3D(
            lat=self.lat,
            lon=self.lon,
            depth=self.depth,
            lat_pos=station_positions["lat"],
            lon_pos=station_positions["lon"],
            depth_pos=station_positions["depth"],
            vel=forward_model.vel_p,
            t_origin=self.t_origin,
        )
        times_s = self.travel_time_3D(
            lat=self.lat,
            lon=self.lon,
            depth=self.depth,
            lat_pos=station_positions["lat"],
            lon_pos=station_positions["lon"],
            depth_pos=station_positions["depth"],
            vel=forward_model.vel_s,
            t_origin=self.t_origin,
        )
        return times_p, times_s

    @staticmethod
    def travel_time_3D(lat, lon, depth, lat_pos, lon_pos, depth_pos, vel, t_origin):
        """
        lat_pos: Station lat locations.

        """
        t = (
            t_origin
            + np.sqrt(
                (lat_pos - lat) ** 2 + (lon_pos - lon) ** 2 + (depth_pos - depth) ** 2
            )
            / vel
        )
        return t
