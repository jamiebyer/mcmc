import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
import pandas as pd
from scipy import interpolate

"""
TODO: 
- generate true model better....
- generate starting models differently than true model (maybe just random bounds)
- actually might wanna try the opposite too.
- fix: failed to find root for fundamental mode
- fix: division by zero
- fix: invalid value in log
"""


class Model:
    def __init__(
        self,
        n_layers,
        n_data,
        freqs,
        param_bounds,
        poisson_ratio,
        density_params,
    ):
        """

        :param freqs: frequencies at which data was collected.
        :param n_data: number of data (phase velocity)
        :param sigma_pd: uncertainty in data (phase velocity)
        :param layer_bounds: min, max bounds for layer thickness.
        :param poisson_ratio:
        :param density_params:
        """

        """
        - input: layer_bounds, poisson, etc.
        - when model created, generate params from bounds
        - create velocity_model from generated params
        - run forward model to get phase_vel from velocity_model

        - ... get params from velocity model
        - perturb params (vel_s, thickness, sigma_pd)
        - generate the rest of params/ velocity model from poisson, etc.

        - either switching between params and velocity, creating new lists, or 
        - mapping between the indices.
        """
        # read csvs
        self.prem = pd.read_csv("./data/PREM500_IDV.csv")

        # *** double check which variables need to be on the class ***
        self.n_layers = n_layers
        self.n_data = n_data
        self.freqs = freqs
        self.periods = 1 / self.freqs

        self.param_bounds = param_bounds
        self.poisson_ratio = poisson_ratio
        self.density_params = density_params

        # assemble model params
        self.model_params = self.generate_model_params()
        self.n_params = len(self.model_params)

    @property
    def layer_bounds(self):
        return self.param_bounds[0]

    @property
    def sigma_pd_bounds(self):
        return self.param_bounds[-1]

    @property
    def thickness(self):
        return self.model_params[: self.n_layers]

    @property
    def vel_s(self):
        return self.model_params[self.n_layers : 2 * self.n_layers]

    @property
    def vel_p(self):
        vp_vs = np.sqrt((2 - 2 * self.poisson_ratio) / (1 - 2 * self.poisson_ratio))
        vel_p = self.vel_s * vp_vs
        return vel_p

    @property
    def density(self):
        # *** don't recompute vel_p ***
        density = (self.vel_p - self.density_params[0]) / self.density_params[1]
        return density

    @property
    def sigma_pd(self):
        return self.model_params[-1]

    def get_thickness(self, model_params):
        return model_params[: self.n_layers]

    def get_vel_s(self, model_params):
        return model_params[self.n_layers : 2 * self.n_layers]

    def get_vel_p(self, vel_s):
        vp_vs = np.sqrt((2 - 2 * self.poisson_ratio) / (1 - 2 * self.poisson_ratio))
        vel_p = vel_s * vp_vs
        return vel_p

    def get_density(self, vel_p):
        density = (vel_p - self.density_params[0]) / self.density_params[1]
        return density

    @staticmethod
    def get_sigma_pd(model_params):
        return model_params[-1]

    def generate_model_params(self):
        """
        generating true velocity model from PREM.

        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        """
        # generate sigma_pd...
        sigma_pd = np.random.uniform(
            self.sigma_pd_bounds[0], self.sigma_pd_bounds[1]
        )  # validate that this is uniform

        # generate layer thicknesses
        thickness = np.random.uniform(
            self.layer_bounds[0], self.layer_bounds[1], self.n_layers
        )  # validate that this is uniform
        # this is setting the depth to be the bottom of each layer ***
        depths = np.cumsum(thickness)
        radius = self.prem['radius[unit="m"]'] / 1000  # m -> km

        # *** initial vel_s is from prem... i guess could do this for the true model and then
        # the others are generated within the bounds? ***
        # get initial vs
        # velocities are split into components
        vsh = self.prem['Vsh[unit="m/s"]'] / 1000  # m/s -> km/s
        vsv = self.prem['Vsv[unit="m/s"]'] / 1000  # m/s -> km/s
        prem_vs = np.sqrt(vsh**2 + vsv**2)
        vs_func = interpolate.interp1d(radius, prem_vs)
        vel_s = vs_func(depths)

        # vel_s = [3.50, 3.40, 3.50, 3.80, 4.20, 4.50, 4.70, 4.80, 4.75, 4.75]

        model_params = np.concatenate((thickness, vel_s, [sigma_pd]))

        return model_params

    def get_birch_params(self):
        radius = self.prem['radius[unit="m"]']
        prem_density = self.prem['density[unit="kg/m^3"]']

        # fit the curve
        density_params = np.polyfit(radius, prem_density, deg=1)
        # returns [-1.91018882e-03  1.46683536e+04]
        return density_params

    def get_velocity_model(self, model_params):
        # thickness, Vp, Vs, density
        # km, km/s, km/s, g/cm3
        thickness = self.get_thickness(model_params)
        vel_s = self.get_vel_s(model_params)
        vel_p = self.get_vel_p(vel_s)
        density = self.get_density(vel_p)

        return [thickness, vel_p, vel_s, density]

    def forward_model(self, model_params):
        """
        Get phase dispersion curve from shear velocities.
        """
        velocity_model = self.get_velocity_model(model_params)
        pd = PhaseDispersion(*velocity_model)

        try:
            pd_rayleigh = pd(self.periods, mode=0, wave="rayleigh")
            # ell = Ellipticity(*velocity_model.T)
            phase_velocity = pd_rayleigh.velocity
            return phase_velocity
        except (DispersionError, ZeroDivisionError) as e:
            # failed to find root for fundamental mode
            # division by zero
            # *** look into these errors and see what kind of parameters are causing them ***
            raise e


class TrueModel(Model):
    def __init__(self, sigma_pd_true, *args):
        """
        Generate true model, which will be used to create simulated observed pd curves.

        :param n_data: Number of observed data to simulate.
        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        :param sigma_pd: Uncertainty to add to simulated data.
        """
        self.sigma_pd_true = sigma_pd_true

        # *** gotta make sure generated model will work with forward model... ***
        # initialize velocity model
        # generates model params and data
        super().__init__(*args)

    def generate_model_params(self):
        """
        generating true velocity model from PREM.

        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        """
        valid_params = False
        # *** i don't want this try-except here. validate without calling forward_model ***
        # *** should know if params are valid before inputing into forward_model ***
        while not valid_params:
            model_params = super().generate_model_params()
            # get simulated true phase dispersion
            try:
                self.phase_vel_true = self.forward_model(model_params)
                valid_params = True
            except (DispersionError, ZeroDivisionError):
                continue

        # *** test values:
        model_params[self.n_layers : 2 * self.n_layers] = [
            3.50,
            3.40,
            3.50,
            3.80,
            4.20,
            4.50,
            4.70,
            4.80,
            4.75,
            4.75,
        ]
        sigma_pd = model_params[-1]  # ***
        # generate simulated observed data by adding noise to true values.
        self.phase_vel_obs = self.phase_vel_true + sigma_pd * np.random.randn(
            self.n_data
        )
        return model_params


class ChainModel(Model):
    def __init__(self, beta, data_obs, n_bins, *args):
        """
        :param beta: inverse temperature; larger values explore less of the parameter space,
        but are more precise; between 0 and 1
        :param data_obs:
        :param n_keep:
        """
        self.data_obs = data_obs

        super().__init__(*args)

        self.beta = beta
        self.sigma_model = 1 / 20  # PC standard deviation
        self.rot_mat = np.eye(self.n_params)  # rotation matrix

        # convergence params
        self.n_cov = 0  # initialize the dividing number for covariance

        self.normalized_model = np.zeros(self.n_params)

        self.mean_model = np.zeros(self.n_params)
        self.mean_model_sum = np.zeros((self.n_params))

        self.cov_mat = np.zeros(
            (self.n_params, self.n_params)
        )  # initialize covariance matrix
        self.cov_mat_sum = np.zeros(
            (self.n_params, self.n_params)
        )  # initialize covariance matrix sum

        # initialize histogram of model parameters
        self.model_hist = np.zeros((n_bins + 1, self.n_params))
        self.bins = np.zeros((n_bins + 1, self.n_params))
        for ind in range(self.n_params - 1):
            # getting bins for the param,
            edge = np.linspace(
                self.param_bounds[ind, 0], self.param_bounds[ind, 1], n_bins + 1
            )
            self.bins[:, ind] = edge

        # acceptance ratio for each parameter
        self.acc_ratio = np.zeros(self.n_params)
        self.swap_acc = 1
        self.swap_prop = 1

    def generate_model_params(self):
        """ """
        valid_params = False
        while not valid_params:
            model_params = super().generate_model_params()
            # get simulated true phase dispersion
            # *** ... if there's an error here, it shouldn't be the starting model... ***
            try:
                self.logL = self.get_likelihood(model_params)  # set model likelihood
                valid_params = True
            except (DispersionError, ZeroDivisionError):
                pass

        return model_params

    def perturb_params(self, scale_factor=1.3):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param bounds: min and max values for each param
        :param phase_vel_obs: observed data, used to calculate likelihood.
        :param scale_factor:
        """
        # normalizing, rotating
        test_params = (self.model_params - self.param_bounds[:, 0]) / self.param_bounds[
            :, 2
        ]
        test_params = np.matmul(np.transpose(self.rot_mat), test_params)

        # *** Cauchy proposal, check other options ***
        # generate params to try; Cauchy proposal
        test_params += (
            scale_factor
            * self.sigma_model
            * np.tan(np.pi * (np.random.rand(len(test_params)) - 0.5))
        )

        # rotating back and scaling again
        test_params = np.matmul(self.rot_mat, test_params)
        test_params = self.param_bounds[:, 0] + (test_params * self.param_bounds[:, 2])

        # boolean array of valid params
        valid_params = (test_params >= self.param_bounds[:, 0]) & (
            test_params <= self.param_bounds[:, 1]
        )

        # loop over params and perturb
        for ind in np.arange(self.n_params)[valid_params]:
            # calculate new likelihood
            try:
                logL_new = self.get_likelihood(
                    test_params,
                )
            except (DispersionError, ZeroDivisionError):
                continue

            # Compute likelihood ratio in log space:
            dlogL = logL_new - self.logL
            # *** ...
            if dlogL == 0:
                continue

            xi = np.random.rand(1)
            # Apply MH criterion (accept/reject)
            if xi <= np.exp(dlogL):
                self.model_params[ind] = test_params[ind]
                self.logL = logL_new

    def get_derivatives(
        self,
        n_sizes,
        size_scale,
        init_step_size_scale,
        phase_vel_diff_bounds,
    ):
        """
        calculate the jacobian for the model

        :param n_sizes: number of step sizes to try
        :param size_scale:
        :param init_step_size_scale:
        :param phase_vel_diff_bounds:
        """
        # propose n_dm=50 step sizes. compute all for all params. find flat section and optimal derivative. add proir estimate to make it stable
        # finding where the derivative is flat to find best / stable value of the derivative for the jacobian.
        # *** go over this again with n_data ***
        step_sizes = np.zeros((self.n_params, n_sizes))
        # set initial step size
        step_sizes[:, 0] = self.model_params * init_step_size_scale
        model_derivatives = np.zeros((self.n_params, n_sizes, self.n_data))

        # Estimate deriv for range of dm values
        for size_ind in range(n_sizes):
            model_pos = self.model_params + step_sizes[:, size_ind]
            model_neg = self.model_params - step_sizes[:, size_ind]

            for param_ind in range(self.n_params):
                # *** double check these conditions... ***
                model_pos[param_ind] = (
                    model_pos[param_ind] + step_sizes[param_ind, size_ind]
                )

                try:
                    phase_vel_pos = self.forward_model(model_pos)
                    phase_vel_neg = self.forward_model(model_neg)

                    # calculate the (derivative) change in phase velocity over change in model params.
                    # unitless difference between positive and negative phase velocities
                    # ***
                    phase_vel_diff = np.abs(
                        (phase_vel_pos - phase_vel_neg)
                        / (phase_vel_pos + phase_vel_neg)
                    )

                    inds = (phase_vel_diff > phase_vel_diff_bounds[0]) & (
                        phase_vel_diff < phase_vel_diff_bounds[1]
                    )
                    np.put(
                        model_derivatives[param_ind, size_ind, :],
                        inds,
                        (
                            # calculating the centered derivative
                            (phase_vel_pos - phase_vel_neg)
                            / (2 * step_sizes[param_ind, size_ind]),
                        ),
                    )
                except (DispersionError, ZeroDivisionError) as e:
                    print(e)
                    # pass

                if size_ind == n_sizes - 1:
                    break
                # setting step sizes for the next loop
                step_sizes[:, size_ind + 1] = step_sizes[:, size_ind] / size_scale

        return model_derivatives

    def get_jacobian(
        self,
        n_sizes=50,
        size_scale=1.5,
        init_step_size_scale=0.1,
        phase_vel_diff_bounds=[1.0e-7, 5],
    ):
        """
        finding the step size where the derivative is best (flattest) (??)
        """
        model_derivatives = self.get_derivatives(
            n_sizes, size_scale, init_step_size_scale, phase_vel_diff_bounds
        )

        # *** also will need to double check this indexing.... ***
        Jac = np.zeros((self.n_data, self.n_params))
        # can prolly simplify these loops ***
        for param_ind in range(self.n_params):
            for data_ind in range(
                self.n_data
            ):  # For each datum, choose best derivative estimate
                best = 1.0e10
                ibest = 1

                for size_ind in range(n_sizes - 2):
                    # check if the derivative will very very large, and set those to a set max value.
                    if np.any(
                        np.abs(
                            model_derivatives[
                                param_ind, data_ind, size_ind : size_ind + 2
                            ]
                        )
                        < 1.0e-7
                    ):
                        test = 1.0e20
                    else:
                        # *** sum of the absolute value of the ......... uhhhh
                        # *** or is it abs of the sum.. :/
                        # i don't think this comparaison is right.
                        test = np.sum(
                            np.abs(
                                (
                                    model_derivatives[
                                        param_ind, data_ind, size_ind : size_ind + 1
                                    ]
                                    / model_derivatives[
                                        param_ind, data_ind, size_ind + 1 : size_ind + 2
                                    ]
                                )
                                / 2
                                - 1
                            )
                        )

                    if (test < best) and (test > 1.0e-7):
                        best = test
                        ibest = size_ind + 1

                Jac[data_ind, param_ind] = model_derivatives[
                    param_ind, data_ind, ibest
                ]  # Best deriv into Jacobian
                if best > 1.0e10:
                    Jac[data_ind, param_ind] = 0.0

        return Jac

    def lin_rot(self, variance):
        """
        making a linear approximation of the rotation matrix and variance for the params.

        :param param_bounds:
        :param sigma_pd: uncertainty in the data
        :param variance: from trial and error?

        :return sigma_pcsd:
        """
        Jac = self.get_jacobian()
        # Scale columns of Jacobian for stability
        # *** validate ***
        Jac = Jac * self.param_bounds[:, 2]  # multiplying by parameter range

        # Uniform bounded priors of width Δmi are approximated by taking C_p to be a diagonal matrix with
        # variances equal to those of the uniform distributions, i.e., ( Δmi ) 12.
        # prior model covariance matrix representing an assumed Gaussian prior density about the current model
        cov_prior_inv = np.diag(self.n_params * [1 / variance])

        # the data covariance matrix
        # *** validate this sigma_pd ***
        cov_data_inv = np.diag(self.n_data * [self.beta / self.sigma_pd**2])

        #
        cov_cur = (
            np.matmul(np.matmul(np.transpose(Jac), cov_data_inv), Jac) + cov_prior_inv
        )

        # singular value decomposition.
        # the singular values, within each vector sorted in descending order.
        # parameter variance in PC space (?)
        rot_mat, s, _ = np.linalg.svd(cov_cur)
        sigma_model = 1 / (2 * np.sqrt(np.abs(s)))  # PC standard deviations

        return sigma_model, rot_mat

    def get_likelihood(self, params):
        """
        :param params: test params to calculate likelihood with
        :param data_obs: the observed data, used to calculate residuals
        """
        # from the velocity model, calculate phase velocity and compare to true data.
        n_params = len(params)  # *** get_likelihood is used in initialization
        try:
            phase_vel_cur = self.forward_model(params)
            sigma_pd = self.get_sigma_pd(params)
            residuals = self.data_obs - phase_vel_cur

            logL = -(1 / 2) * n_params * np.log(sigma_pd) - np.sum(residuals**2) / (
                2 * sigma_pd**2
            )
            return np.sum(logL)

        except (DispersionError, ZeroDivisionError) as e:
            raise e

    def update_model_hist(self):
        """
        updating the hist for this model, which stores parameter values from all the models

        :param param_bounds:
        """
        for ind in range(self.n_params):
            # getting bins for the param,
            edge = self.bins[:, ind]
            idx_diff = np.argmin(abs(edge - self.model_params[ind]))
            self.model_hist[idx_diff, ind] += 1

    def update_covariance_matrix(self, update_rot_mat):
        """
        :param param_bounds: min, max, range of params; used to normalize model params.
        """

        # *** covariance matrix is a running sum... collecting from burn in stage. cov mat linear estrimate during burn in, switch to running
        # sum at end of burn in and then keep static? could keep updating and diminish the effect ***

        # *** rename params ***

        # can we get the covariance matrix for both chains at the same time?
        # does the numpy cov function help?

        # these variables could just be local..? does it matter?

        # *** make sure there are no nans in output matrix ***

        # normalizing
        self.normalized_model = (
            self.model_params - self.param_bounds[:, 0]
        ) / self.param_bounds[:, 2]

        self.mean_model_sum += self.normalized_model  # calculating the sum of mean(m)
        self.n_cov += 1  # number of covariance matrices in the sum

        # *** this doesn't seem quite right... ****
        self.mean_model = self.mean_model_sum / self.n_cov
        self.cov_mat_sum = self.cov_mat_sum + np.outer(
            np.transpose(self.normalized_model - self.mean_model),
            self.normalized_model - self.mean_model,
        )

        # calculating covariance matrix from samples
        self.cov_mat = self.cov_mat_sum / self.n_cov

        # *** simplify ***
        # dividing the covariance matrix by the auto-correlation of the params, and data
        for row in range(self.n_params):
            for col in range(self.n_params):
                self.cov_mat[row, col] /= np.sqrt(  # invalid scalar divide
                    self.cov_mat[row, row] * self.cov_mat[col, col]
                )
        # after burn in is over, update covariance rot_mat and s from cov_mat
        if update_rot_mat:
            self.rot_mat, s, _ = np.linalg.svd(
                self.cov_mat
            )  # rotate it to its Singular Value Decomposition
            self.sigma_model = np.sqrt(s)
            print(self.sigma_model)
