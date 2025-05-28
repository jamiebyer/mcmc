import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from fk_processing.run_geopsy import get_dispersion_curve

np.complex_ = np.complex64


class Data:
    def __init__(self, periods, data_obs, sigma_data):
        self.periods = periods
        self.data_obs = data_obs
        self.n_data = len(self.data_obs)
        self.sigma_data = sigma_data

        if len(sigma_data) == self.n_data:
            self.data_cov = np.diag(sigma_data**2)
        elif len(sigma_data) == 1:
            self.data_cov = np.eye(self.n_data) * sigma_data


class FieldData(Data):
    def __init__(self, path):
        periods, phase_vels, stds = self.read_observed_data(path)

        super().__init__(periods, phase_vels, stds)

    def read_observed_data(self, path):
        """
        read dispersion curve
        """
        freqs, phase_vels, stds = get_dispersion_curve(path)
        periods = 1 / freqs
        # sort

        return periods, phase_vels, stds


class SyntheticData(Data):
    def __init__(self, periods, sigma_data, thickness, vel_s, vel_p, density):
        velocity_model = np.array([thickness + [0], vel_p, vel_s, density])
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


class Model:

    def __init__(
        self,
        n_layers: int,
        poisson_ratio: float,
        sigma_model,
        beta,
        n_bins,
    ):
        """
        :param beta: inverse temperature; larger values explore less of the parameter space,
            but are more precise; between 0 and 1
        :param n_bins: number of bins for histogram
        """

        self.n_layers = n_layers

        # used for computing model params
        self.poisson_ratio = poisson_ratio
        self.sigma_model = sigma_model

        # assemble model params
        self.n_params = 2 * n_layers - 1
        self._thickness = None
        self._vel_s = None
        self._vel_p = None
        self._density = None

        self.logL = None
        self.data_pred = None

        self.beta = beta

        # variables for storing and computing covariance matrix after burn-in
        self.rot_mat = np.eye(self.n_params)  # initialize rotation matrix
        self.n_cov = 0  # initialize the dividing number for covariance
        self.mean_model = np.zeros(self.n_params)
        self.mean_model_sum = np.zeros((self.n_params))

        # initialize covariance matrix
        self.cov_mat = np.zeros((self.n_params, self.n_params))
        self.cov_mat_sum = np.zeros((self.n_params, self.n_params))

        # acceptance ratio for each parameter
        self.swap_acc = 0
        self.swap_prop = 0
        self.swap_err = 0

        # initialize histogram of model parameters
        self.n_bins = n_bins
        self.model_hist = np.zeros((self.n_params, n_bins + 1))

    @property
    def layer_bounds(self):
        return self.param_bounds[0]

    @property
    def thickness(self):
        # return self.model_params[: (self.n_layers - 1)]
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        # validate bounds
        # self.model_params[: (self.n_layers - 1)] = thickness
        self._thickness = thickness

    @property
    def vel_s(self):
        # return self.model_params[self.n_layers : 2 * self.n_layers]
        return self._vel_s

    @vel_s.setter
    def vel_s(self, vel_s):
        # validate bounds
        # self.model_params[self.n_layers : 2 * self.n_layers] = vel_s
        self._vel_s = vel_s

        # update vel_p and density
        self.vel_p = self.get_vel_p(self._vel_s)
        self.density = self.get_density(self.vel_p)

    @property
    def vel_p(self):
        # check bounds
        return self._vel_p

    @vel_p.setter
    def vel_p(self, vel_p):
        # check bounds
        self._vel_p = vel_p

    @property
    def density(self):
        # check bounds
        return self._density

    @density.setter
    def density(self, density):
        # check bounds
        self._density = density

    def get_vel_p(self, vel_s):
        vp_vs = np.sqrt((2 - 2 * self.poisson_ratio) / (1 - 2 * self.poisson_ratio))
        vel_p = vel_s * vp_vs
        return vel_p

    def get_density(self, vel_p):
        density = (1741 * np.sign(vel_p) * abs(vel_p) ** (1 / 4)) / 1000
        return density

    @staticmethod
    def assemble_param_bounds(bounds, n_layers):
        # reshape bounds to be the same shape as params
        param_bounds = np.concatenate(
            (
                [bounds["thickness"]] * (n_layers - 1),
                [bounds["vel_s"]] * n_layers,
                [bounds["vel_p"]] * n_layers,
                [bounds["density"]] * n_layers,
                # [bounds["sigma_model"]],
            ),
            axis=0,
        )

        # add the range of the bounds to param_bounds as a third column (min, max, range)
        range = param_bounds[:, 1] - param_bounds[:, 0]
        param_bounds = np.column_stack((param_bounds, range))

        return param_bounds

    def get_velocity_model(self, param_bounds, thickness, vel_s):
        vel_p = self.get_vel_p(vel_s)
        density = self.get_density(vel_p)
        velocity_model = np.array([list(thickness) + [0], vel_p, vel_s, density])

        # validate bounds and physics...
        valid_thickness = (thickness >= param_bounds["thickness"][0]) & (
            thickness <= param_bounds["thickness"][1]
        )
        valid_vel_s = (vel_s >= param_bounds["vel_s"][0]) & (
            vel_s <= param_bounds["vel_s"][1]
        )
        valid_vel_p = (vel_p >= param_bounds["vel_p"][0]) & (
            vel_p <= param_bounds["vel_p"][1]
        )
        valid_density = (density >= param_bounds["density"][0]) & (
            density <= param_bounds["density"][1]
        )

        valid_params = np.all(
            valid_thickness & valid_vel_s & valid_vel_p & valid_density
        )

        return velocity_model, valid_params

    def forward_model(self, periods, velocity_model):
        """
        get phase dispersion curve for current shear velocities and layer thicknesses.

        :param model_params: model params to use to get phase dispersion
        *** generalize later***
        """
        # *** keep track of errors in forward model
        # get phase dispersion curve
        # thickness, Vp, Vs, density
        # km, km/s, km/s, g/cm3
        pd = PhaseDispersion(*velocity_model)

        # try calculating phase_velocity from given params.
        try:
            pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
            # ell = Ellipticity(*velocity_model.T)
            phase_velocity = pd_rayleigh.velocity

            return phase_velocity
        except (DispersionError, ZeroDivisionError) as e:
            # *** errors: ***
            # failed to find root for fundamental mode
            # division by zero
            raise e

    def generate_model_params(self, param_bounds):
        """
        generating initial params for new model.
        """
        # only for thickness and shear velocity, compute for vel_p and density
        model_params = np.random.uniform(
            param_bounds[:, 0], param_bounds[:, 1], self.n_params
        )

        return model_params

    def get_optimization_model(
        self,
        param_bounds,
        data,
        T_0=100,
        epsilon=0.95,
        ts=100,
        n_steps=1000,
    ):
        """
        :param T_0: initial temp
        :param epsilon: decay factor of temperature
        :param ts: number of temperature steps
        :param n_steps: number of balancing steps at each temp
        """

        temps = np.zeros(ts)
        temps[0] = T_0

        for k in range(1, ts):
            temps[k] = temps[k - 1] * epsilon

        # temps = np.ones(ts)

        results = {
            "temps": [],  # temps,
            # "params": [],
            "thickness": [],
            "vel_s": [],
            "vel_p": [],
            "density": [],
            "logL": [],
        }

        # reduce logarithmically
        for T in temps:
            print("\ntemp", T)
            # number of steps at this temperature
            for _ in range(n_steps):
                print(_)
                # perturb each parameter
                # if using the perturb params function, it does the acceptance rate stats in the function
                self.perturb_params(param_bounds, data, T=T)

                results["temps"].append(T)
                results["thickness"].append(self.thickness.copy())
                results["vel_s"].append(self.vel_s.copy())
                results["vel_p"].append(self.vel_p.copy())
                results["density"].append(self.density.copy())
                results["logL"].append(self.logL)

        df = pd.DataFrame(results)
        df.to_csv("./results/inversion/optimize_model.csv")

    def perturb_params(
        self,
        param_bounds,
        data,
        rotation=False,
        T=None,
        sample_prior=False,
        prior_dist="cauchy",
    ):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param scale_factor:
        """
        # should be validating params, generate until have valid values
        # get bounds in rotated space? validate in rotated space...

        if rotation:
            # normalizing params
            # double check ***
            norm_params = (self.model_params - param_bounds[:, 0]) / param_bounds[:, 2]
            # rotating params
            rotated_params = np.matmul(np.transpose(self.rot_mat), norm_params)
            rotated_bounds = np.matmul(
                np.transpose(self.rot_mat), param_bounds[:, :1] - param_bounds[:, 0]
            )

            # validate params

            # rotating back
            perturbed_norm_params = np.matmul(self.rot_mat, perturbed_rotated_params)
            # rescaling
            perturbed_params = param_bounds[:, 0] + (
                perturbed_norm_params * param_bounds[:, 2]
            )

        # thickness_scale, vel_s_scale = 10, 100
        thickness_scale, vel_s_scale = 0.001, 0.001
        # thickness_scale, vel_s_scale = 1, 1
        # loop over params and perturb each individually
        for ind in range(len(self.thickness)):
            test_thickness = self.thickness.copy()
            # uniform distribution
            if prior_dist == "uniform":
                test_thickness[ind] += (
                    (np.random.uniform() - 0.5)
                    * thickness_scale
                    * self.sigma_model["thickness"]
                    * (param_bounds["thickness"][1] - param_bounds["thickness"][0])
                    / 2
                )
            elif prior_dist == "cauchy":
                # cauchy distribution
                test_thickness[ind] += (
                    self.sigma_model["thickness"]
                    * thickness_scale
                    * np.tan(np.pi * (np.random.uniform() - 0.5))
                )

            test_velocity_model, valid_params = self.get_velocity_model(
                param_bounds, test_thickness, self.vel_s.copy()
            )

            # check acceptance criteria
            acc = self.acceptance_criteria(
                test_velocity_model, valid_params, data, sample_prior=sample_prior, T=T
            )
            if acc:
                self.thickness = test_thickness.copy()
                self.swap_acc += 1
            elif not sample_prior:
                self.swap_prop += 1

        for ind in range(len(self.vel_s)):
            test_vel_s = self.vel_s.copy()
            if prior_dist == "uniform":
                # uniform distribution
                test_vel_s[ind] += (
                    (np.random.uniform() - 0.5)
                    * vel_s_scale
                    * self.sigma_model["vel_s"]
                    * (param_bounds["vel_s"][1] - param_bounds["vel_s"][0])
                    / 2
                )
            elif prior_dist == "cauchy":
                # cauchy distribution
                test_vel_s[ind] += (
                    self.sigma_model["vel_s"]
                    * vel_s_scale
                    * np.tan(np.pi * (np.random.uniform() - 0.5))
                )

            test_velocity_model, valid_params = self.get_velocity_model(
                param_bounds, self.thickness.copy(), test_vel_s
            )

            # check acceptance criteria
            acc = self.acceptance_criteria(
                test_velocity_model, valid_params, data, sample_prior=sample_prior, T=T
            )
            if acc:
                self.vel_s = test_vel_s.copy()
                self.swap_acc += 1
            elif not sample_prior:
                self.swap_prop += 1

    def acceptance_criteria(
        self, test_velocity_model, valid_params, data, T, sample_prior
    ):
        if not valid_params:
            return False
        if sample_prior:
            logL_new = 1
            return True

        try:
            logL_new, data_pred_new = self.get_likelihood(test_velocity_model, data)
        except (DispersionError, ZeroDivisionError):
            # add specific condition here for failed forward model
            self.swap_err += 1
            return False

        # Compute likelihood ratio in log space:
        dlogL = logL_new - self.logL
        if T is not None:
            dlogL = dlogL / T

        xi = np.random.rand(1)
        # Apply MH criterion (accept/reject)
        if dlogL < 0 or xi <= np.exp(-dlogL):
            self.swap_acc += 1
            self.logL = logL_new
            self.data_pred = data_pred_new
            return True
        else:
            self.swap_prop += 1
            return False

    def get_derivatives(
        self,
        periods,
        n_sizes,
        data_diff_bounds,
    ):
        """
        calculate the jacobian for the model

        :param n_sizes: number of step sizes to try
        :param phase_vel_diff_bounds:
        """
        # propose n_dm=50 step sizes. compute all for all params. find flat section and optimal derivative. add prior
        # estimate to make it stable finding where the derivative is flat to find best / stable value of the derivative
        # for the jacobian.

        # step size is scaled from the param range
        # size_scale = 1.5
        # init_step_size_scale = 0.1
        step_scales = np.linspace(0.1, 0.001, n_sizes)

        step_sizes = np.repeat(step_scales, self.n_params) * self.model_params[:, 2]

        model_derivatives = np.zeros((self.n_params, self.n_data, n_sizes))

        # estimate deriv for range of dm values
        for param_ind in range(self.n_params):
            model_pos = self.model_params + step_sizes[param_ind, :]
            model_neg = self.model_params - step_sizes[param_ind:, :]

            model_pos[param_ind] = model_pos[param_ind] + step_sizes[param_ind, :]

            try:
                data_pos = self.forward_model(periods, model_pos)
                data_neg = self.forward_model(periods, model_neg)

                # calculate the change in phase velocity over change in model param
                # unitless difference between positive and negative phase velocities
                data_diff = np.abs((data_pos - data_neg) / (data_pos + data_neg))

                # calculate centered derivative for values with reasonable differences(?)
                inds = (data_diff > data_diff_bounds[0]) & (
                    data_diff < data_diff_bounds[1]
                )
                model_derivatives[:, :, inds] = (data_pos - data_neg) / (
                    2 * step_sizes[param_ind, inds]
                )
            except (DispersionError, ZeroDivisionError) as e:
                pass

        return model_derivatives

    def get_jacobian(
        self,
    ):
        """
        finding the step size where the derivative is stable (flat)
        """
        n_sizes = 50
        phase_vel_diff_bounds = [1.0e-7, 5]

        model_derivatives = self.get_derivatives(
            n_sizes, phase_vel_diff_bounds
        )  # [param, deriv, data]

        Jac = np.zeros((self.n_data, self.n_params))

        # get indices of derivatives that are too small
        small_indices = []
        large_indices = []
        best_indices = []
        for s in range(n_sizes - 2):
            small_indices.append(
                np.any(np.abs(model_derivatives[:, s : s + 2, :]) < 1.0e-7)
            )
            large_indices.append(
                np.any(np.abs(model_derivatives[:, s : s + 2, :]) > 1.0e10)
            )
            # want three in a row
            # smallest difference between them?
            # absolute value of the sum of the left and right derivatives

            flatness = np.sum(model_derivatives[:, s : s + 2, :])
            best = np.argmin(flatness)
            best_indices.append(model_derivatives[:, best, :])

        Jac = model_derivatives[:, best_indices, :]

        return Jac

    def linearized_rotation(self, param_bounds):
        """
        making a linear approximation of the rotation matrix and variance for the params.

        :param variance: from the uniform distribution/ prior

        :return sigma_pcsd:
        """
        Jac = self.get_jacobian()
        # Scale columns of Jacobian for stability
        Jac = Jac * self.param_bounds[:, 2]  # multiplying by parameter range

        # Uniform bounded priors of width Δmi are approximated by taking C_p to be a diagonal matrix with
        cov_prior_inv = np.diag(self.n_params * [1 / param_bounds[:, 2]])

        # the data covariance matrix
        cov_data_inv = np.diag(self.n_data * [self.beta / self.sigma_data**2])

        cov_cur = (
            np.matmul(np.matmul(np.transpose(Jac), cov_data_inv), Jac) + cov_prior_inv
        )

        # parameter variance in PC space (?)
        rot_mat, s, _ = np.linalg.svd(cov_cur)
        sigma_model = 1 / (2 * np.sqrt(np.abs(s)))  # PC standard deviations

        return rot_mat, sigma_model

    def get_likelihood(self, velocity_model, data):
        """
        :param model_params: params to calculate likelihood with
        """
        # return 1, periods
        try:
            data_pred = self.forward_model(data.periods, velocity_model)
            residuals = data.data_obs - data_pred
            # for identical errors
            # logL = np.sum(residuals**2) / (self.sigma_model**2)
            logL = (residuals**2).T @ (1 / data.data_cov) @ (data.sigma_data**2)

            return logL, data_pred

        except (DispersionError, ZeroDivisionError) as e:
            raise e

    def update_model_hist(self):
        """
        updating the hist for this model, which stores parameter values from all the models
        """
        # The bins for this hist should be the param bounds
        for ind in range(self.n_params):
            counts, bins = np.histogram(x, n_bins)

            # getting bins for the param,
            # edge = self.bins[:, ind]
            # idx_diff = np.argmin(abs(edge - self.model_params[ind]))
            # self.model_hist[idx_diff, ind] += 1

    def update_rotation_matrix(self, burn_in):
        # for burn in period, update rotation matrix by linearization
        # after burn in, start saving samples in covariance matrix
        # after burn in (and cov mat stabilizes) start using cov mat to get rotation matrix
        # update covariance matrix

        if burn_in:
            # linearize
            rot_mat, sigma_model = self.linearized_rotation(self.param_bounds)
        else:
            rot_mat, sigma_model = self.update_covariance_matrix()

        self.rot_mat, self.sigma_model = rot_mat, sigma_model

    def update_covariance_matrix(self):
        """ """
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

        rot_mat, s, _ = np.linalg.svd(
            self.cov_mat
        )  # rotate it to its Singular Value Decomposition
        sigma_model = np.sqrt(s)

        # s is the step size? sampling based on sigma_model

        return rot_mat, sigma_model
