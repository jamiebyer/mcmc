import numpy as np
from disba import PhaseDispersion
import pandas as pd
from scipy import interpolate


class Model:
    def __init__(self, freqs, n_data, thickness, vel_p, vel_s, sigma_pd):
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
        self.velocity_model = [self.thickness, self.vel_p, self.vel_s, self.density]

        # map params indices to velocity model indices.
        # n_layers = len(self.thickness)
        # params_indices = np.arange(self.n_params)

        # velocity_model_indices = np.concatenate((np.arange(n_layers), np.arange(2*n_layers, 3*n_layers)))
        # self.map_indices = dict(zip(params_indices, velocity_model_indices))

    @property
    def velocity_model():
        return self.get_velocity_model()

    # abstract function
    def get_velocity_model():
        pass

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
    def __init__(self, layer_bounds, poisson_ratio, density_params, sigma_pd):
        """
        Generate true model, which will be used to create simulated observed pd curves.

        :param n_data: Number of observed data to simulate.
        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        :param sigma_pd: Uncertainty to add to simulated data.
        """
        self.initialize_model(layer_bounds, poisson_ratio, density_params, sigma_pd)

        # initialize velocity model
        super().__init__(thickness, vel_p, vel_s, sigma_pd)

        # generate simulated data and observations for the true model
        self.create_simulated_data()

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

    def initialize_model(self, layer_bounds, poisson_ratio, density_params, sigma_pd):
        """
        generating true velocity model from PREM.

        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        :param sigma_pd: Uncertainty to add to simulated data.
        """

        # from PREM...
        prem = pd.read_csv("./PREM500_IDV.csv")

        # generate layer thicknesses
        thickness = np.random.uniform(
            layer_bounds[0], layer_bounds[1], self.n_data
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

        self.vel_s = vel_s
        self.vel_p = vel_p
        self.density = density
        self.thickness = thickness

    def get_birch_params():
        # fit to prem
        # density = (vel_p - density_params[0]) / density_params[1]
        prem = pd.read_csv("./PREM500_IDV.csv")

        radius = prem['radius[unit="m"]']
        prem_density = prem['density[unit="kg/m^3"]']

        # fit the curve
        density_params = np.polyfit(radius, prem_density, deg=1)
        # returns [-1.91018882e-03  1.46683536e+04]


class ChainModel(Model):
    def __init__(self, thickness, vel_p, vel_s, sigma_pd, n_keep, n_bins=200):
        super().__init__(thickness, vel_p, vel_s, sigma_pd)

        self.vel_phase = None
        self.logL = None

        self.pcsd = 1 / 20  # PC standard deviation
        self.u = np.eye(self.n_params)

        # convergence params
        # *** rename ***
        self.ncov = 0  # initialize the dividing number for covariance

        self.mbar = np.zeros(self.n_params)
        self.mmsum = np.zeros(
            (self.n_params)
        )  # initiazlize parameter mean vector: becareful with the dimension of Nx1 vs just N (a vector)
        self.cov_mat_sum = np.zeros(
            (self.n_params, self.n_params)
        )  # initialize covariance matrix sum
        self.cov_mat = np.zeros(
            (self.n_params, self.n_params)
        )  # initialize covariance matrix, 3 dimmensions with Nchain

        self.hist_m = np.zeros(
            (n_bins + 1, self.n_params)
        )  # initialize histogram of model parameter, 10 bins -> 11 edges by Npar

        saved_results = {
            "logL": np.zeros(n_keep),
            "m": np.zeros(n_keep),
            "d": np.zeros(n_keep),
            "acc": np.zeros(n_keep),
        }

    def validate_params(params, param_bounds):
        return np.all(params >= param_bounds[:, 0]) and np.all(
            params <= param_bounds[:, 1]
        )

    def generate_starting_models(
        n_chains, freqs, true_model, param_bounds, sigma_pd, pcsd=0.05
    ):
        """
        Loop until a valid starting model is created.

        """
        n_params = true_model.n_params
        chains = []
        for c in range(n_chains):
            valid_model = False
            # maybe there's another way to generate starting model
            starting_model = true_model.copy()
            true_params = true_model.params

            while not valid_model:
                starting_params = true_params + pcsd * np.random.randn(
                    true_model.n_params
                )
                # validate, check bounds
                valid_model = ChainModel.validate_params(starting_params, param_bounds)

            starting_model.params = starting_params
            # calculate likelihood
            starting_model.logL = ChainModel.get_likelihood(
                freqs,
                starting_model.velocity_model,
                sigma_pd,
                n_params,
                true_model.phase_vel_obs,
            )
            chains.append(starting_model)  # validate

        return chains

    def perturb_params(self, param_bounds, phase_vel_obs, scale_factor=1.3):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param bounds: min and max values for each param
        :param phase_vel_obs: observed data, used to calculate likelihood.
        :param scale_factor:
        """

        # normalizing, rotating
        test_params = (self.params - param_bounds[:, 0]) / param_bounds[:, 1]
        test_params = np.matmul(np.transpose(u), test_params)

        # *** Cauchy proposal, check other options ***
        # generate params to try; Cauchy proposal
        test_params += (
            scale_factor
            * self.pcsd
            * np.tan(np.pi * (np.random.rand(len(test_params)) - 0.5))
        )

        # rotating back and scaling again
        test_params = np.matmul(self.u, test_params)
        test_params = param_bounds[:, 0] + (test_params * param_bounds[:, 1])

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

    def get_jacobian(self, freqs, n_dm=50):
        """
        n_freqs is also n_depths | n_layers
        """
        n_freqs = len(freqs)
        dm_start = self.params * 0.1
        dm = np.zeros(n_dm, self.n_params)
        dm[0] = dm_start
        dRdm = np.zeros((self.n_params, n_freqs, n_dm))

        # Estimate deriv for range of dm values
        for dm_ind in range(n_dm):
            model_pos = self.params + dm[dm_ind]
            model_neg = self.params - dm[dm_ind]

            for param_ind in range(self.n_params):
                model_pos[param_ind] = model_pos[param_ind] + dm[param_ind, dm_ind]

                # each frequency is a datum?
                pdr_pos = VelocityModel.forward_model(freqs, model_pos)
                pdr_neg = VelocityModel.forward_model(freqs, model_neg)

                dRdm[param_ind, :, dm_ind] = np.where(
                    (np.abs((pdr_pos - pdr_neg) / (pdr_pos + pdr_neg)) > 1.0e-7 < 5),
                    (pdr_pos - pdr_neg) / (2.0 * dm[param_ind, dm_ind]),
                    dRdm[param_ind, :, dm_ind],
                )

                # setting dm for the next loop
                if dm_ind < n_dm - 1:
                    dm[:, dm_ind + 1] = dm[:, dm_ind] / 1.5

        Jac = self.get_best_derivative(dRdm, n_freqs, n_dm)

        return Jac

    def get_best_derivative(self, dRdm, n_freqs, n_dm):
        Jac = np.zeros((n_freqs, self.n_params))
        # can prolly simplify these loops ***
        for param_ind in range(self.n_params):
            for freq_ind in range(
                n_freqs
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

    def lin_rot(self, freqs, param_bounds, sigma=0.1):
        n_freqs = len(freqs)

        Jac = self.get_jacobian(freqs, n_freqs)
        Jac = Jac * param_bounds[:, 2]  # Scale columns of Jacobian for stability

        # what should this value be ??
        Mpriorinv = np.diag(self.n_params * [12])  # variance for U[0,1]=1/12

        Cdinv = np.diag(1 / sigma**2)  # what should sigma be ??

        JactCdinv = np.matmul(np.transpose(Jac), Cdinv)
        Ctmp = np.matmul(JactCdinv, Jac) + Mpriorinv

        V, L, VT = np.linalg.svd(Ctmp)
        pcsd = 0.5 * (1.0 / np.sqrt(np.abs(L)))  # PC standard deviations

        return pcsd

    def get_likelihood(freqs, velocity_model, sigma_pd, n_params, phase_vel_obs):
        """ """
        # from the velocity model, calculate phase velocity and compare to true data.
        phase_vel_cur = Model.forward_model(freqs, velocity_model)

        residuals = phase_vel_obs - phase_vel_cur

        logL = -(1 / 2) * n_params * np.log(sigma_pd) - np.sum(residuals**2) / (
            2 * sigma_pd**2
        )

        return np.sum(logL)

    def update_hist(self):
        for ipar in np.arange(len(mt)):
            edge = np.linspace(minlim[ipar], maxlim[ipar], nbin + 1)
            idx_dif = np.argmin(abs(edge - mcur[ipar, ichain]))
            self.hist_m[idx_dif, ipar, ichain] += 1

    def update_covariance_matrix(self, n_params, param_bounds, rotation: bool):
        """
        np.cov: A 1-D or 2-D array containing multiple variables and observations.
        Each row of m represents a variable, and each column a single observation of all those variables.
        """

        # *** rename params ***

        # can we get the covariance matrix for both chains at the same time?
        # does the numpy cov function help?

        # normalizing
        mw = (self.params - param_bounds[:, 0]) / param_bounds[:, 2]

        self.mmsum += mw  # calculating the sum of mean(m)
        self.ncov += 1
        self.mbar = self.mmsum / self.ncov
        self.mcsum = self.mcsum + np.outer(np.transpose(mw - self.mbar), mw - self.mbar)
        self.cov_mat = self.mcsum / self.ncov  # calculating covariance matrix

        for ipar in range(n_params):
            for jpar in range(n_params):
                self.cov_mat[ipar, jpar] = self.cov_mat[ipar, jpar] / np.sqrt(
                    self.cov_mat[ipar, ipar] * self.cov_mat[jpar, jpar]
                )
        # rotation
        if rotation:
            self.u, s, vh = np.linalg.svd(
                self.cov_mat
            )  # rotate it to its Singular Value Decomposition
            self.pcsd = np.sqrt(s)
