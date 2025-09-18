import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
from inversion.model import Model

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

    def get_data_dict(self):
        data_dict = {
            "coords": {
                "period": {"dims": ["period"], "data": self.periods},
            },
            "data_vars": {
                "data_obs": {"dims": ["period"], "data": self.data_obs},
            },
            "attrs": {"sigma_data": self.sigma_data},
        }

        return data_dict


class FieldData(Data):
    def __init__(self, periods, phase_vels, stds):
        # periods, phase_vels, stds = self.read_observed_data(path)

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
    def __init__(self, periods, noise, model_params_obj, depth, vel_s):
        """ """
        data_true, data_obs, sigma_data, model_params = self.generate_observed_data(
            periods, noise, model_params_obj, depth, vel_s
        )
        self.data_true = data_true
        self.model_true = model_params

        super().__init__(periods, data_obs, sigma_data)

        # get true likelihood
        self.logL_true = Model.get_likelihood(self, data_true, sigma_data)

    def generate_true_model(self, periods, noise, bounds, n_layers):
        """
        generating random model
        """
        valid_params = False
        while not valid_params:
            depth = np.random.uniform(
                bounds["depth"][0],
                bounds["depth"][1],
                n_layers - 1,
            )
            vel_s = np.random.uniform(bounds["vel_s"][0], bounds["vel_s"][1], n_layers)
            vel_p = np.random.uniform(bounds["vel_p"][0], bounds["vel_p"][1], n_layers)
            density = np.random.uniform(
                bounds["density"][0], bounds["density"][1], n_layers
            )
            velocity_model = np.array([list(depth) + [0], vel_p, vel_s, density])

            try:
                pd = PhaseDispersion(*velocity_model)
                pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
                valid_params = True
            except (DispersionError, ZeroDivisionError) as e:
                pass

        data_true = pd_rayleigh.velocity
        sigma_data = noise * data_true
        # sigma_data is a percentage, so multiply by true data
        data_obs = data_true + sigma_data * np.random.randn(len(periods))
        model_true = np.concatenate((depth, vel_s))

        return data_true, data_obs, sigma_data, model_true

    def generate_observed_data(self, periods, noise, model_params_obj, depth, vel_s):
        model_params = np.array(depth + vel_s)
        # use forward_model function
        data_true = model_params_obj.forward_model(periods, model_params)

        # sigma_data is a percentage, so multiply by true data
        sigma_data = noise * data_true
        data_obs = data_true + sigma_data * np.random.randn(len(periods))
        return data_true, data_obs, sigma_data, model_params

    def get_data_dict(self):
        data_dict = super().get_data_dict()
        data_dict["data_vars"].update(
            {
                "data_true": {"dims": ["period"], "data": self.data_true},
                "model_true": {"dims": ["n_model_params"], "data": self.model_true},
            }
        )
        data_dict["attrs"].update({"logL_true": self.logL_true})
        return data_dict
