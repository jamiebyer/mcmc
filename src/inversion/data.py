import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

np.complex_ = np.complex64


class Data:
    def __init__(self, periods, data_obs, sigma_data):
        self.periods = periods
        self.data_obs = data_obs
        self.n_data = len(self.data_obs)
        self.sigma_data = sigma_data

        if isinstance(sigma_data, list) and len(sigma_data) == self.n_data:
            self.data_cov = np.diag(sigma_data**2)
        elif isinstance(sigma_data, float):
            self.data_cov = np.eye(self.n_data) * sigma_data**2


class FieldData(Data):
    def __init__(self, path):
        periods, phase_vels, stds = self.read_observed_data(path)

        super().__init__(periods, phase_vels, stds)

    def read_observed_data(self, path):
        """
        read dispersion curve
        """
        freqs, phase_vels, stds = None, None, None  # get_dispersion_curve(path)
        periods = 1 / freqs
        # sort

        return periods, phase_vels, stds


class SyntheticData(Data):
    def __init__(self, periods, sigma_data, model_params, thickness, vel_s):
        params = np.array(thickness + vel_s)
        velocity_model = model_params.get_velocity_model(params)
        data_true, data_obs = self.generate_observed_data(
            periods, sigma_data, velocity_model
        )
        self.data_true = data_true
        super().__init__(periods, data_obs, sigma_data)

    def generate_observed_data(self, periods, sigma_data, velocity_model):
        pd = PhaseDispersion(*velocity_model)
        pd_rayleigh = pd(periods, mode=0, wave="rayleigh")

        data_true = pd_rayleigh.velocity
        data_obs = data_true + sigma_data * np.random.randn(len(periods))
        return data_true, data_obs


class GeneratedData(Data):
    def __init__(self, periods, sigma_data, bounds, n_layers):
        # velocity_model = np.array([thickness + [0], vel_p, vel_s, density])
        data_true, data_obs = self.generate_observed_data(
            periods, sigma_data, bounds, n_layers
        )
        self.data_true = data_true
        super().__init__(periods, data_obs, sigma_data)

    def generate_observed_data(self, periods, sigma_data, bounds, n_layers):
        valid_params = False
        while not valid_params:
            thickness = np.random.uniform(
                bounds["thickness"][0],
                bounds["thickness"][1],
                n_layers - 1,
            )
            vel_s = np.random.uniform(bounds["vel_s"][0], bounds["vel_s"][1], n_layers)
            vel_p = np.random.uniform(bounds["vel_p"][0], bounds["vel_p"][1], n_layers)
            density = np.random.uniform(
                bounds["density"][0], bounds["density"][1], n_layers
            )
            velocity_model = np.array([list(thickness) + [0], vel_p, vel_s, density])

            try:
                pd = PhaseDispersion(*velocity_model)
                pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
                valid_params = True
            except (DispersionError, ZeroDivisionError) as e:
                pass

        data_true = pd_rayleigh.velocity
        data_obs = data_true + sigma_data * np.random.randn(len(periods))
        return data_true, data_obs
