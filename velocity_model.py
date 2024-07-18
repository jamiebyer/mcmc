import numpy as np
from disba import PhaseDispersion
import pandas as pd
from scipy import interpolate


class Model:
    def __init__(self, freqs, n_data, vel_s, vel_p, density, thickness, sigma_pd):
        self.freqs = freqs
        self.n_data = n_data

        # should velocity model and params be properties?
        self.thickness = thickness
        self.vel_p = vel_p
        self.vel_s = vel_s
        self.sigma_pd = sigma_pd

        self.params = np.concatenate((self.thickness, self.vel_s, [self.sigma_pd]))
        self.n_params = len(self.params)
        self.n_data = self.n_params  # * difference between n params and n data?
        # self.velocity_model = [self.thickness, self.vel_p, self.vel_s, self.density]

    @property
    def velocity_model(self):
        # return [self.thickness, self.vel_p, self.vel_s, self.density]
        return self.get_velocity_model()

    # abstract function
    def get_velocity_model(self):
        return [self.thickness, self.vel_p, self.vel_s, self.density]

    @staticmethod
    def generate_model_params(n_data, layer_bounds, poisson_ratio, density_params):
        """
        generating true velocity model from PREM.

        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        """

        # from PREM...
        prem = pd.read_csv("./PREM500_IDV.csv")

        # generate layer thicknesses
        thickness = np.random.uniform(
            layer_bounds[0], layer_bounds[1], n_data
        )  # validate that this is uniform
        # this is setting the depth to be the bottom of each layer ***
        depth = np.cumsum(thickness)
        radius = prem['radius[unit="m"]'] / 1000  # m -> km

        # """
        # interpolate density
        # prolly want the avg of each layer or something ***
        prem_density = prem['density[unit="kg/m^3"]'] / 1000  # kg/m^3 -> g/cm3
        density_func = interpolate.interp1d(radius, prem_density)
        density = density_func(depth)
        # """

        # get initial vs
        # velocities are split into components
        vsh = prem['Vsh[unit="m/s"]'] / 1000  # m/s -> km/s
        vsv = prem['Vsv[unit="m/s"]'] / 1000  # m/s -> km/s
        prem_vs = np.sqrt(vsh**2 + vsv**2)
        vs_func = interpolate.interp1d(radius, prem_vs)
        vel_s = vs_func(depth)

        # get initial vp
        vp_vs = np.sqrt((2 - 2 * poisson_ratio) / (1 - 2 * poisson_ratio))
        vel_p = vel_s * vp_vs

        # or since it's the true model, use prem vp
        # vph = prem['Vph[unit="m/s"]'] / 1000  # m/s -> km/s
        # vpv = prem['Vpv[unit="m/s"]'] / 1000  # m/s -> km/s
        # prem_vp = np.sqrt(vph**2 + vpv**2)
        # vp_func = interpolate.interp1d(radius, prem_vp)
        # vp = vp_func(depth)

        # Birch's law
        # self.density = (vel_p - density_params[0]) / density_params[1]

        return vel_s, vel_p, density, thickness

    def get_birch_params():
        # fit to prem
        # density = (vel_p - density_params[0]) / density_params[1]
        prem = pd.read_csv("./PREM500_IDV.csv")

        radius = prem['radius[unit="m"]']
        prem_density = prem['density[unit="kg/m^3"]']

        # fit the curve
        density_params = np.polyfit(radius, prem_density, deg=1)
        # returns [-1.91018882e-03  1.46683536e+04]
        return density_params

    def forward_model(freqs, model):
        """
        Get phase dispersion curve from shear velocities.
        """
        periods = 1 / freqs
        pd = PhaseDispersion(*model.velocity_model)
        pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
        # ell = Ellipticity(*velocity_model.T)

        return pd_rayleigh


class TrueModel(Model):
    def __init__(self, *args):
        """
        Generate true model, which will be used to create simulated observed pd curves.

        :param n_data: Number of observed data to simulate.
        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        :param sigma_pd: Uncertainty to add to simulated data.
        """
        # initialize velocity model
        super().__init__(*args)
        # generate simulated data and observations for the true model
        self.generate_simulated_data()

    def generate_simulated_data(self):
        """
        generate simulated data and observations.
        """
        # get simulated true phase dispersion
        pd_rayleigh = self.forward_model(self.freqs, self.velocity_model)
        # generate simulated observed data by adding noise to true values.
        self.phase_vel_true = pd_rayleigh.velocity
        # *** the true sigma_pd on the model should be generated? it's not the same as initial guess for the model. ***
        self.phase_vel_obs = self.phase_vel_true + self.sigma_pd * np.random.randn(
            self.n_data
        )


class ChainModel(Model):
    def __init__(self, beta, *args):
        """
        :param beta:
        """
        super().__init__(*args)

        self.beta = beta

        self.logL = None
        self.pcsd = 1 / 20  # PC standard deviation
        self.u = np.eye(self.n_params)

        # convergence params
        self.ncov = 0  # initialize the dividing number for covariance

        self.mw = np.zeros(self.n_params)
        self.mbar = np.zeros(self.n_params)
        self.mmsum = np.zeros((self.n_params))  # initialize parameter mean vector
        self.cov_mat_sum = np.zeros(
            (self.n_params, self.n_params)
        )  # initialize covariance matrix sum
        self.cov_mat = np.zeros(
            (self.n_params, self.n_params)
        )  # initialize covariance matrix
        self.hist_m = None

        # acceptance ratio for each parameter
        self.acc_ratio = np.zeros(self.n_params)
        self.swap_acc = 1
        self.swap_prop = 1

        self.saved_results = {
            "logL": np.zeros(n_keep),
            "m": np.zeros(n_keep),
            "d": np.zeros(n_keep),
            "acc": np.zeros(n_keep),
        }

    def validate_params(params, param_bounds):
        return np.all(params >= param_bounds[:, 0]) and np.all(
            params <= param_bounds[:, 1]
        )

    def perturb_params(self, param_bounds, phase_vel_obs, scale_factor=1.3):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param bounds: min and max values for each param
        :param phase_vel_obs: observed data, used to calculate likelihood.
        :param scale_factor:
        """

        # normalizing, rotating
        test_params = (self.params - param_bounds[:, 0]) / param_bounds[:, 2]
        test_params = np.matmul(np.transpose(self.u), test_params)

        # *** Cauchy proposal, check other options ***
        # generate params to try; Cauchy proposal
        test_params += (
            scale_factor
            * self.pcsd
            * np.tan(np.pi * (np.random.rand(len(test_params)) - 0.5))
        )

        # rotating back and scaling again
        test_params = np.matmul(self.u, test_params)
        test_params = param_bounds[:, 0] + (test_params * param_bounds[:, 2])

        # *** this is incorrect... ***
        # loop over params and perturb
        for ind in range(self.n_params):
            # validate test params
            if not ChainModel.validate_params(test_params[ind], param_bounds):
                continue

            # calculate new likelihood
            # update velocity model with test value
            # get ind that maps params and velocity model
            self.params = np.concatenate((self.thickness, self.vel_s, [self.sigma_pd]))
            self.velocity_model = [self.thickness, self.vel_p, self.vel_s, self.density]

            logL_new = ChainModel.get_likelihood(
                self.freqs,
                self.velocity_model,
                self.sigma_pd,
                self.n_params,
                phase_vel_obs,
            )

            # Compute likelihood ratio in log space:
            dlogL = logL_new - self.logL
            xi = np.random.rand(1)
            # Apply MH criterion (accept/reject)
            if xi <= np.exp(dlogL):
                self.logL = logL_new
                self.params = test_params  # validate this
                # self.vel_rayleigh =

    def get_jacobian(self, n_dm=50):
        """ """
        # *** go over this again with n_data ***
        dm_start = self.params * 0.1
        dm = np.zeros(n_dm, self.n_params)
        dm[0] = dm_start
        dRdm = np.zeros((self.n_params, self.n_data, n_dm))

        # Estimate deriv for range of dm values
        for dm_ind in range(n_dm):
            model_pos = self.params + dm[dm_ind]
            model_neg = self.params - dm[dm_ind]

            for param_ind in range(self.n_params):
                model_pos[param_ind] = model_pos[param_ind] + dm[param_ind, dm_ind]

                # each frequency is a datum?
                pdr_pos = Model.forward_model(self.freqs, model_pos)
                pdr_neg = Model.forward_model(self.freqs, model_neg)

                dRdm[param_ind, :, dm_ind] = np.where(
                    (np.abs((pdr_pos - pdr_neg) / (pdr_pos + pdr_neg)) > 1.0e-7 < 5),
                    (pdr_pos - pdr_neg) / (2.0 * dm[param_ind, dm_ind]),
                    dRdm[param_ind, :, dm_ind],
                )

                # setting dm for the next loop
                if dm_ind < n_dm - 1:
                    dm[:, dm_ind + 1] = dm[:, dm_ind] / 1.5

        Jac = self.get_best_derivative(dRdm, self.n_data, n_dm)

        return Jac

    def get_best_derivative(self, dRdm, n_dm):
        Jac = np.zeros((self.n_data, self.n_params))
        # can prolly simplify these loops ***
        for param_ind in range(self.n_params):
            for freq_ind in range(
                self.n_data
            ):  # For each datum, choose best derivative estimate
                best = 1.0e10
                ibest = 1

                for dm_ind in range(n_dm - 2):
                    # check if the derivative will very very large
                    if np.any(
                        np.abs(dRdm[param_ind, freq_ind, dm_ind : dm_ind + 2]) < 1.0e-7
                    ):
                        test = 1.0e20
                    else:
                        test = np.abs(
                            np.sum(
                                (
                                    dRdm[param_ind, freq_ind, dm_ind : dm_ind + 1]
                                    / dRdm[param_ind, freq_ind, dm_ind + 1 : dm_ind + 2]
                                )
                                / 2
                                - 1
                            )
                        )

                    if (test < best) and (test > 1.0e-7):
                        best = test
                        ibest = dm_ind + 1

                Jac[freq_ind, param_ind] = dRdm[
                    param_ind, freq_ind, ibest
                ]  # Best deriv into Jacobian
                if best > 1.0e10:
                    Jac[freq_ind, param_ind] = 0.0

        return Jac

    def lin_rot(self, param_bounds, sigma_pd, variance=1 / 12):
        """
        :param param_bounds:
        :param sigma_pd: uncertainty in the data
        :param variance: from trial and error?

        :return sigma_pcsd:
        """
        Jac = self.get_jacobian()
        # Scale columns of Jacobian for stability
        # *** validate ***
        Jac = Jac * param_bounds[:, 2]  # multiplying by parameter range

        # prior model covariance matrix representing an assumed Gaussian prior density about the current model
        cov_prior_inv = np.diag(self.n_params / variance)

        # the data covariance matrix
        cov_data_inv = np.diag(self.beta / sigma_pd**2)

        #
        cov_cur = (
            np.matmul(np.matmul(np.transpose(Jac), cov_data_inv), Jac) + cov_prior_inv
        )

        # singular value decomposition.
        # the singular values, within each vector sorted in descending order.
        # parameter variance in PC space (?)
        _, L, _ = np.linalg.svd(cov_cur)
        sigma_pcsd = 1 / (2 * np.sqrt(np.abs(L)))  # PC standard deviations

        return sigma_pcsd, L

    def get_likelihood(freqs, velocity_model, sigma_pd, n_params, phase_vel_obs):
        """ """
        # from the velocity model, calculate phase velocity and compare to true data.
        phase_vel_cur = Model.forward_model(freqs, velocity_model)

        residuals = phase_vel_obs - phase_vel_cur

        logL = -(1 / 2) * n_params * np.log(sigma_pd) - np.sum(residuals**2) / (
            2 * sigma_pd**2
        )

        return np.sum(logL)

    def update_hist(self, param_bounds, n_bins):
        """ """
        # *** not sure where n_bins should be stored. maybe hist_m should be a property ***
        if self.hist_m is None:
            self.hist_m = np.zeros(
                (n_bins + 1, self.n_params)
            )  # initialize histogram of model parameters

        for ind in range(self.n_params):
            edge = np.linspace(param_bounds[ind, 0], param_bounds[ind, 1], n_bins + 1)
            idx_diff = np.argmin(abs(edge - self.params[ind]))
            self.hist_m[idx_diff, ind] += 1

    def update_covariance_matrix(self, param_bounds):
        """
        np.cov: A 1-D or 2-D array containing multiple variables and observations.
        Each row of m represents a variable, and each column a single observation of all those variables.
        """

        # *** rename params ***

        # can we get the covariance matrix for both chains at the same time?
        # does the numpy cov function help?

        # these variables could just be local..? does it matter?

        # normalizing
        self.m_w = (self.params - param_bounds[:, 0]) / param_bounds[:, 2]

        self.mm_sum += self.m_w  # calculating the sum of mean(m)
        self.n_cov += 1  #

        self.m_bar = self.mm_sum / self.n_cov
        self.mc_sum = self.mc_sum + np.outer(
            np.transpose(self.mw - self.m_bar), self.m_w - self.m_bar
        )
        self.cov_mat = self.mc_sum / self.n_cov  # calculating covariance matrix

        for param_ind in range(self.n_params):
            for data_ind in range(self.n_data):
                self.cov_mat[param_ind, data_ind] = self.cov_mat[
                    param_ind, data_ind
                ] / np.sqrt(
                    self.cov_mat[param_ind, param_ind]
                    * self.cov_mat[data_ind, data_ind]
                )
