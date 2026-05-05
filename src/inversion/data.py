import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
from inversion.model import Model

import matplotlib.pyplot as plt

np.complex_ = np.complex64


class Data:
    def __init__(self, periods, data_obs, noise_dist, noise_params):
        self.periods = periods
        self.data_obs = data_obs
        self.n_data = len(self.data_obs)
        self.noise_dist = noise_dist
        self.noise_params = noise_params

        # if isinstance(sigma_data, list) and len(sigma_data) == self.n_data:
        #     self.data_cov = np.diag(sigma_data**2)
        # elif isinstance(sigma_data, float):
        #     self.data_cov = np.eye(self.n_data) * sigma_data**2

    def get_data_dict(self):
        data_dict = {
            "coords": {
                "period": {"dims": ["period"], "data": self.periods},
            },
            "data_vars": {
                "data_obs": {"dims": ["period"], "data": self.data_obs},
            },
            "attrs": {"noise_dist": self.noise_dist},
        }

        for k, v in self.noise_params.items():
            data_dict["attrs"][k] = v

        return data_dict


class FieldData(Data):
    def __init__(self, periods, phase_vels, stds):
        super().__init__(periods, phase_vels, stds)


class SyntheticData(Data):
    def __init__(
        self, periods, noise_dist, noise_params, model_params_obj, depth, vel_s
    ):
        """ """
        data_true, data_obs, sigma_data, model_params = self.generate_observed_data(
            periods, noise_dist, noise_params, model_params_obj, depth, vel_s
        )
        self.data_true = data_true
        self.model_true = model_params

        super().__init__(periods, data_obs, noise_dist, noise_params)

        # get true likelihood
        noise_params["noise_percent"] = sigma_data
        self.logL_true = Model.get_likelihood(self, data_true, noise_dist, noise_params)

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

    def generate_observed_data(
        self, periods, noise_dist, noise_params, model_params_obj, depth, vel_s
    ):
        model_params = np.array(depth + vel_s)
        # use forward_model function
        data_true = model_params_obj.forward_model(periods, model_params)

        noise_percent = noise_params["noise_percent"]
        if noise_dist == "normal":
            sigma_data = noise_percent * data_true
            # sigma_data is a percentage, so multiply by true data
            data_obs = data_true + sigma_data * np.random.randn(len(periods))

        elif noise_dist == "asym-laplace":
            AL_q_5_list, AL_q_95_list = [], []
            norm_q_5_list, norm_q_95_list = [], []

            lambd, kappa = noise_params["lambd"], noise_params["kappa"]
            sigma_data = noise_percent * data_true

            mu = 0
            lambd = (1 / (3.5 * sigma_data)) * lambd

            x = np.linspace(-50, 50, 100000)

            noise = []
            for ind in range(len(data_true)):
                s = np.sign(x - mu)
                pdf = (lambd[ind] / (kappa + 1 / kappa)) * np.exp(
                    -(x - mu) * lambd[ind] * s * kappa**s
                )

                norm_pdf = (1 / np.sqrt(2 * np.pi * sigma_data[ind] ** 2)) * np.exp(
                    -((x - mu) ** 2 / (2 * sigma_data[ind] ** 2))
                )

                # integrate distribution
                # the cdf should go from 0 to 1
                dx = x[1] - x[0]
                cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)
                norm_cdf = np.cumsum(((norm_pdf[:-1] + norm_pdf[1:]) / 2) * dx)

                q_5 = x[np.argmin(np.abs(cdf - 0.05))]
                q_95 = x[np.argmin(np.abs(cdf - 0.95))]
                AL_q_5_list.append(q_5)
                AL_q_95_list.append(q_95)

                q_5 = x[np.argmin(np.abs(norm_cdf - 0.05))]
                q_95 = x[np.argmin(np.abs(norm_cdf - 0.95))]
                norm_q_5_list.append(q_5)
                norm_q_95_list.append(q_95)

                # generate a random uniform number between 0 and 1
                n = np.random.uniform(0, 1)

                # use to select value from inverse of cdf
                ind = np.argmin(np.abs(cdf - n))
                x_pick = (x[ind] + x[ind + 1]) / 2

                noise.append(x_pick)

            data_obs = data_true + noise
        """
        plt.scatter(periods, data_true)
        plt.scatter(periods, data_obs)

        plt.plot(periods, data_true + np.array(AL_q_5_list), c="red")
        plt.plot(periods, data_true + np.array(AL_q_95_list), c="red")

        plt.plot(periods, data_true + np.array(norm_q_5_list), c="orange")
        plt.plot(periods, data_true + np.array(norm_q_95_list), c="orange")

        plt.show()
        """
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
