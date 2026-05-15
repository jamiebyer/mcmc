import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
from inversion.model import Model

import matplotlib.pyplot as plt
from matplotlib import cm

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

        # for k, v in self.noise_params.items():
        #     data_dict["attrs"][k] = v

        return data_dict


class FieldData(Data):
    def __init__(self, periods, phase_vels, stds):
        super().__init__(periods, phase_vels, stds)


class SyntheticData(Data):
    def __init__(
        self, periods, noise_dist, noise_params, model_params_obj, depth, vel_s
    ):
        """ """
        # generate true data
        model_params = np.array(depth + vel_s)
        data_true = model_params_obj.forward_model(periods, model_params)

        self.model_true = model_params
        self.data_true = data_true

        # for frequency dependent noise, scale using the true data
        if noise_params["frequency_scaling"]:
            if noise_dist == "normal":
                noise_params["std"] = noise_params["std_percent"] * data_true
            elif noise_dist == "asym-laplace":
                noise_params["lambd_scale"] = (
                    noise_params["lambd_scale_percent"] * data_true
                )

        # generate observed data
        data_obs = self.generate_observed_data(periods, noise_dist, noise_params)

        super().__init__(periods, data_obs, noise_dist, noise_params)

        # get true likelihood
        # need to get the likelihood using normal assumption and AL
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

    def generate_observed_data(self, periods, noise_dist, noise_params):
        """
        generate observed data by generating random noise and adding it to the true data.
        """

        # can give the noise frequency-dependent scaling using either
        # a percent of the true data
        # or an exponential based on values from fitting the spread/percentiles of the field data

        if noise_dist == "normal":
            std_data = noise_params["std"]
            data_obs = self.data_true + std_data * np.random.randn(len(periods))

        elif noise_dist == "asym-laplace":
            # AL_q_5_list, AL_q_95_list = [], []
            # norm_q_5_list, norm_q_95_list = [], []

            # get lambda scaling from noise_params
            lambd, kappa = noise_params["lambd"], noise_params["kappa"]
            lambd_scaling = noise_params["lambd_scale"]

            mu = 0
            lambd = (1 / lambd_scaling) * lambd

            # for each frequency, define the pdf for the error distribution
            # integrate to get the cdf
            # and pick noise using the cdf
            x = np.linspace(-50, 50, 100000)
            noise = []
            for ind in range(len(self.data_true)):
                if isinstance(lambd, float):
                    l = lambd
                else:
                    l = lambd[ind]

                s = np.sign(x - mu)
                pdf = (l / (kappa + 1 / kappa)) * np.exp(-(x - mu) * l * s * kappa**s)

                # integrate distribution
                # the cdf should go from 0 to 1
                dx = x[1] - x[0]
                cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)

                # generate a random uniform number between 0 and 1
                n = np.random.uniform(0, 1)

                # use to select value from inverse of cdf
                i = np.argmin(np.abs(cdf - n))
                x_pick = (x[i] + x[i + 1]) / 2

                noise.append(x_pick)

            data_obs = self.data_true + noise

        return data_obs

    @staticmethod
    def get_noise_pdf(noise_dist, noise_params, freq_ind=None, mu=0):
        x = np.linspace(-50, 50, 100000)

        if noise_dist == "normal":
            std_data = noise_params["std"]

            if isinstance(std_data, float):
                std = std_data
            else:
                std = std_data[freq_ind]

            pdf = (1 / np.sqrt(2 * np.pi * std**2)) * np.exp(
                -((x - mu) ** 2 / (2 * std**2))
            )

        if noise_dist == "asym-laplace":
            lambd, kappa = noise_params["lambd"], noise_params["kappa"]
            lambd_scaling = noise_params["lambd_scale"]

            lambd = (1 / lambd_scaling) * lambd

            if isinstance(lambd, float):
                l = lambd
            else:
                l = lambd[freq_ind]

            s = np.sign(x - mu)
            pdf = (l / (kappa + 1 / kappa)) * np.exp(-(x - mu) * l * s * kappa**s)

        # integrate distribution
        # the cdf should go from 0 to 1
        dx = x[1] - x[0]
        cdf = np.cumsum(((pdf[:-1] + pdf[1:]) / 2) * dx)

        # q_low, q_high = 0.159, 0.841
        q_5 = x[np.argmin(np.abs(cdf - 0.05))]
        q_95 = x[np.argmin(np.abs(cdf - 0.95))]

        return x, pdf, cdf, q_5, q_95

    def generate_noise_dist(self):
        # can give the noise frequency-dependent scaling using either
        # a percent of the true data
        # or an exponential based on values from fitting the spread/percentiles of the field data

        # lower: 15.9, higher: 84.1, to have 68.2 range
        AL_q_lower_list, AL_q_higher_list = [], []
        norm_q_lower_list, norm_q_higher_list = [], []

        freqs_2d, noise_2d = [], []
        stds = []
        for ind in range(len(self.data_true)):
            x, pdf, cdf, q_5, q_95 = self.get_noise_pdf(
                self.noise_dist, self.noise_params, freq_ind=ind
            )
            AL_q_lower_list.append(q_5)
            AL_q_higher_list.append(q_95)

            picks = []
            for _ in range(10000):
                # generate a random uniform number between 0 and 1
                n = np.random.uniform(0, 1)

                # use to select value from inverse of cdf
                i = np.argmin(np.abs(cdf - n))
                x_pick = (x[i] + x[i + 1]) / 2

                picks.append(self.data_true[ind] + x_pick)

            std = np.std(picks)
            stds.append(std)

            x, pdf, cdf, q_5, q_95 = SyntheticData.get_noise_pdf(
                noise_dist="normal", noise_params={"std": std}, freq_ind=ind
            )
            norm_q_lower_list.append(q_5)
            norm_q_higher_list.append(q_95)

            freqs_2d += len(picks) * [1 / self.periods[ind]]
            noise_2d += picks

        stds = np.array(stds)

        return (
            freqs_2d,
            noise_2d,
            AL_q_lower_list,
            AL_q_higher_list,
            norm_q_lower_list,
            norm_q_higher_list,
            stds,
        )

    def plot_simulated_data_frequencies(
        self,
        freqs_2d,
        noise_2d,
        AL_q_lower,
        AL_q_higher,
        norm_q_lower,
        norm_q_higher,
        stds,
    ):
        # at each frequency
        # plot asymmetric laplacian
        # plot the normal distribution using the mode and the standard deviation of the data
        freq_bins = np.logspace(
            np.log10(np.min(freqs_2d)),
            np.log10(np.max(freqs_2d)),
            len(self.periods) + 1,
        )
        noise_bins = np.linspace(np.min(noise_2d), np.max(noise_2d), 150)
        # plt.hist2d(freqs_2d, noise_2d, bins=[freq_bins, noise_bins])

        # plot 2d histogram with normalizing each column/frequency
        hist, xedges, yedges = np.histogram2d(
            freqs_2d, noise_2d, bins=[freq_bins, noise_bins]
        )
        hist = hist.T
        hist *= 1 / hist.sum(axis=0, keepdims=True)
        hist[hist == 0] = np.nan

        for ind, p in enumerate(self.periods):
            freq = 1 / p
            inds_2d = np.isclose(freqs_2d, np.repeat(freq, len(freqs_2d)))

            plt.clf()

            plt.axvline(self.data_true[ind], c="red", label="data true")
            plt.axvline(self.data_obs[ind], c="red", ls="--", label="data obs")

            # plot hist of noise at this frequency
            plt.hist(np.array(noise_2d)[inds_2d], bins=30, density=True)

            # plot AL distribution
            x, pdf, cdf, q_5, q_95 = self.get_noise_pdf(
                self.noise_dist, self.noise_params, mu=self.data_true[ind], freq_ind=ind
            )
            plt.plot(x, pdf, label="AL")
            plt.axvline(q_5, c="black")
            plt.axvline(q_95, c="black")

            # plot normal distribution
            x, pdf, cdf, q_5, q_95 = SyntheticData.get_noise_pdf(
                noise_dist="normal",
                noise_params={"std": stds[ind]},
                mu=self.data_true[ind],
                freq_ind=ind,
            )
            plt.plot(x, pdf, label="normal")
            plt.axvline(q_5, c="black", ls="--")
            plt.axvline(q_95, c="black", ls="--")

            plt.legend()

            plt.xlim([-0.1, 1.5])

            plt.title(
                "freq: "
                + str(np.round(freq, 2))
                # + "\nkappa: "
                # + str(self.noise_params["kappa"])
                # + ", lambda: "
                # + str(self.noise_params["lambd"])
            )

            # plt.show()
            plt.savefig(
                "./figures/simulated_data/freqs/hist-" + str(np.round(freq, 2)) + ".png"
            )

    def plot_simulated_data_hist2d(
        self,
        freqs_2d,
        noise_2d,
        AL_q_lower,
        AL_q_higher,
        norm_q_lower,
        norm_q_higher,
        stds,
    ):
        # scaled by exponential
        # (OR by a percentage of the dispersion curve)
        # loop over frequencies
        # generate 10000 points from asymmetric laplacian distribution
        plt.clf()
        freq_bins = np.logspace(
            np.log10(np.min(freqs_2d)),
            np.log10(np.max(freqs_2d)),
            len(self.periods) + 1,
        )
        noise_bins = np.linspace(np.min(noise_2d), np.max(noise_2d), 150)
        # plt.hist2d(freqs_2d, noise_2d, bins=[freq_bins, noise_bins])

        # plot 2d histogram with normalizing each column/frequency
        hist, xedges, yedges = np.histogram2d(
            freqs_2d, noise_2d, bins=[freq_bins, noise_bins]
        )
        hist = hist.T
        hist *= 1 / hist.sum(axis=0, keepdims=True)
        hist[hist == 0] = np.nan
        mesh = plt.pcolormesh(xedges, yedges, hist)

        # show true data, generated observed data
        """
        plt.errorbar(
            1 / self.periods - 0.01,
            self.data_true,
            yerr=[-np.array(AL_q_lower), AL_q_higher],
            label="AL errors",
            ls="none",
            c="lightgrey",
            elinewidth=3,
            zorder=3,
        )
        plt.errorbar(
            1 / self.periods + 0.01,
            self.data_true,
            yerr=[-np.array(norm_q_lower), norm_q_higher],
            label="norm errors",
            ls="none",
            c="dimgrey",
            elinewidth=3,
            zorder=3,
        )
        """
        plt.scatter(
            1 / self.periods,
            self.data_true,
            label="data true",
            c="darkgrey",
            edgecolors="black",
            zorder=4,
        )
        plt.scatter(
            1 / self.periods,
            self.data_obs,
            label="data obs",
            c="white",
            edgecolors="black",
            zorder=4,
        )

        plt.ylim([0, 1.5])
        plt.xscale("log")

        plt.xlabel("frequency (Hz)")
        plt.ylabel("velocity (km/s)")

        # plt.title(
        #     "kappa: "
        #     + str(self.noise_params["kappa"])
        #     + ", lambda: "
        #     + str(self.noise_params["lambd"])
        # )

        plt.legend()
        plt.colorbar(mesh)

        # plt.show()
        plt.savefig("./figures/simulated_data/hist2d.png")

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
