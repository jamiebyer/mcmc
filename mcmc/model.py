import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
import pandas as pd
from scipy import interpolate

"""
TODO: 
- change how the true model and starting models are generated.
- fix: failed to find root for fundamental mode
- fix: division by zero
- fix: invalid value in log
- generalize layer_bounds
"""


class Model:
    def __init__(
        self,
        n_layers: int,
        n_data: int,
        freqs: np.ndarray,
        param_bounds,
        poisson_ratio: float,
        density_params: list,
    ):
        """
        :param n_layers: number of layers in the model.
        :param n_data: number of data collected
        :param freqs: frequencies at which data was collected.
        :param sigma_model: uncertainty in data
        :param param_bounds: array of (min, max, range) for each parameter. same length as n_params
        :param poisson_ratio: value for poisson's ratio used to approximate vel_p from vel_s
        :param density_params: birch params used to estimate the density profile of the model
        """
        self.n_layers = n_layers
        self.n_data = n_data
        self.periods = 1 / freqs

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
    def sigma_model_bounds(self):
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
    def sigma_model(self):
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
    def get_sigma_model(model_params):
        return model_params[-1]

    def generate_model_params(self):
        """
        generating initial params for new model.
        """
        # can add other information to make a better initial model.
        # make sure generating true model is different than method for generating starting models for chains
        # could add a starting estimate for sigma_model

        # *** set vel_s to increase velocity with depth (genneral) ***
        model_params = np.random.uniform(
            self.param_bounds[:, 0], self.param_bounds[:, 1], self.n_layers
        )

        return model_params

    def get_velocity_model(self, model_params):
        """
        not used for generalized inversion.
        reshape model params to be inputed into forward model PhaseDispersion.
        """
        # thickness, Vp, Vs, density
        # km, km/s, km/s, g/cm3
        thickness = self.get_thickness(model_params)
        vel_s = self.get_vel_s(model_params)
        vel_p = self.get_vel_p(vel_s)
        density = self.get_density(vel_p)

        return [thickness, vel_p, vel_s, density]

    def forward_model(self, model_params):
        """
        not general. modify the forward model for the type of inversion.
        get phase dispersion curve for current shear velocities and layer thicknesses.

        :param model_params: model params to use to get phase dispersion
        """
        # get phase dispersion curve
        velocity_model = self.get_velocity_model(model_params)
        pd = PhaseDispersion(*velocity_model)

        # try calculating phase_velocity from given params.
        try:
            pd_rayleigh = pd(self.periods, mode=0, wave="rayleigh")
            # ell = Ellipticity(*velocity_model.T)
            phase_velocity = pd_rayleigh.velocity
            return phase_velocity
        except (DispersionError, ZeroDivisionError) as e:
            # *** errors: ***
            # failed to find root for fundamental mode
            # division by zero
            raise e


class TrueModel(Model):
    def __init__(self, *args):
        """
        Generate true model, which will be used to create simulated observed pd curves.

        :param n_data: Number of observed data to simulate.
        :param layer_bounds: [min, max] for layer thicknesses. (m)
        :param poisson_ratio:
        :param density_params: Birch params to simulate density profile.
        """

        super().__init__(*args)  # generates model params and data

    def generate_model_params(self):
        """
        generating true velocity model.
        """

        # *** currently the forward model errors a lot. i want to put better constraints on the params
        #     so there aren't so many errors. i don't like this try-except, i want to validate the
        #     params without calling the forward model. ***

        # generating initial model params. generate params until the forward model runs without error.
        valid_params = False
        while not valid_params:
            model_params = super().generate_model_params()
            try:
                # get the true data values for the true model
                self.phase_vel_true = self.forward_model(model_params)
                valid_params = True
            except (DispersionError, ZeroDivisionError):
                continue

        # generate simulated observed data by adding noise to true values.
        # *** validate the random distribution ***
        self.phase_vel_obs = self.phase_vel_true + self.sigma_model * np.random.randn(
            self.n_data
        )
        return model_params


class ChainModel(Model):
    def __init__(self, beta, data_obs, n_bins, *args):
        """
        :param beta: inverse temperature; larger values explore less of the parameter space,
            but are more precise; between 0 and 1
        :param data_obs: data observations- used to calculate likelihood of the model.
        :param n_bins: number of bins for histogram ***
        """
        self.data_obs = data_obs

        super().__init__(*args)

        self.beta = beta
        self.rot_mat = np.eye(self.n_params)  # initialize rotation matrix

        self.n_cov = 0  # initialize the dividing number for covariance

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
        # self.acc_ratio = np.zeros(self.n_params)
        self.swap_acc = 0
        self.swap_prop = 0

    def generate_model_params(self):
        """
        generate starting model parameters. calculate likelihood of the parameters.
        loop until the parameters are valid and the forward_model doesn't fail.
        """
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

    def perturb_params(self, scale_factor):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param scale_factor:
        """
        # normalizing, rotating
        test_params = (self.model_params - self.param_bounds[:, 0]) / self.param_bounds[
            :, 2
        ]
        test_params = np.matmul(np.transpose(self.rot_mat), test_params)

        # *** Cauchy proposal, check other options ***
        # *** figure out what the scale_factor is.. ***
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
            if dlogL == 0:
                continue

            xi = np.random.rand(1)
            # Apply MH criterion (accept/reject)
            if xi <= np.exp(dlogL):
                self.swap_acc += 1
                self.model_params[ind] = test_params[ind]
                self.logL = logL_new
            else:
                self.swap_prop += 1

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
        # propose n_dm=50 step sizes. compute all for all params. find flat section and optimal derivative. add prior
        # estimate to make it stable finding where the derivative is flat to find best / stable value of the derivative
        # for the jacobian.
        # *** validate, go over this again with n_data ***
        step_sizes = np.zeros((self.n_params, n_sizes))
        # set initial step size
        step_sizes[:, 0] = self.model_params * init_step_size_scale
        model_derivatives = np.zeros((self.n_params, n_sizes, self.n_data))

        # estimate deriv for range of dm values
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

                    # calculate the change in phase velocity over change in model param
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
                    pass

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
        :param sigma_model: uncertainty in the data
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
        # *** validate this sigma_model ***
        cov_data_inv = np.diag(self.n_data * [self.beta / self.sigma_model**2])

        cov_cur = (
            np.matmul(np.matmul(np.transpose(Jac), cov_data_inv), Jac) + cov_prior_inv
        )

        # singular value decomposition.
        # the singular values, within each vector sorted in descending order.
        # parameter variance in PC space (?)
        rot_mat, s, _ = np.linalg.svd(cov_cur)
        sigma_model = 1 / (2 * np.sqrt(np.abs(s)))  # PC standard deviations

        return sigma_model, rot_mat

    def get_likelihood(self, test_params):
        """
        :param params: test params to calculate likelihood with
        :param data_obs: the observed data, used to calculate residuals
        """
        # from the velocity model, calculate phase velocity and compare to true data.
        n_params = len(test_params)  # *** get_likelihood is used in initialization
        try:
            phase_vel_cur = self.forward_model(test_params)
            sigma_model = self.get_sigma_model(test_params)
            residuals = self.data_obs - phase_vel_cur

            logL = -(1 / 2) * n_params * np.log(sigma_model) - np.sum(
                residuals**2
            ) / (2 * sigma_model**2)
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
        :param update_rot_mat: whether or not to update the rotation matrix. this also updates sigma_model.
            after the burn-in, we switch to calculating the rotation matrix from the covariance matrix.
        """
        # normalizing
        normalized_model = (
            self.model_params - self.param_bounds[:, 0]
        ) / self.param_bounds[:, 2]

        self.mean_model_sum += normalized_model  # calculating the sum of mean
        self.n_cov += 1  # number of covariance matrices in the sum

        # *** validate this ****
        mean_model = self.mean_model_sum / self.n_cov
        self.cov_mat_sum = self.cov_mat_sum + np.outer(
            np.transpose(normalized_model - mean_model),
            normalized_model - mean_model,
        )

        # calculating covariance matrix from samples
        self.cov_mat = self.cov_mat_sum / self.n_cov

        # *** simplify? ***
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
