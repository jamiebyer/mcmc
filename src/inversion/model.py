import numpy as np
from disba._exception import DispersionError
import pandas as pd

np.complex_ = np.complex64


class Model:

    def __init__(
        self,
        model_params,
        beta=None,
    ):
        """
        :param beta: inverse temperature; larger values explore less of the parameter space,
            but are more precise; between 0 and 1
        """
        # initialize

        self.logL = None
        self.data_pred = None

        # acceptance ratio for each parameter
        self.swap_acc = 0
        self.swap_rej = 0
        self.swap_err = 0

        self.beta = beta

        """
        # variables for storing and computing covariance matrix after burn-in
        self.rot_mat = np.eye(self.n_params)  # initialize rotation matrix
        self.n_cov = 0  # initialize the dividing number for covariance
        self.mean_model = np.zeros(self.n_params)
        self.mean_model_sum = np.zeros((self.n_params))
        # initialize covariance matrix
        self.cov_mat = np.zeros((self.n_params, self.n_params))
        self.cov_mat_sum = np.zeros((self.n_params, self.n_params))
        
        # initialize histogram of model parameters
        self.n_bins = n_bins
        self.model_hist = np.zeros((self.n_params, n_bins + 1))
        """

    @staticmethod
    def validate_bounds(param_bounds, model_params):
        """
        validate bounds.
        """
        valid_params = np.all(
            (model_params >= param_bounds[0]) & (model_params <= param_bounds[1])
        )
        return valid_params

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
        proposal_distribution,
        scale_factor,
        rotation=False,
        T=None,
        sample_prior=False,
    ):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param scale_factor:
        """

        if rotation:
            """
            # normalizing params
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
            """
            pass

        thickness_scale, vel_s_scale = scale_factor
        # loop over params and perturb each individually
        # *** currently hard-coded for velocity and thickness. generalize later;
        # remove duplicate code. ***

        # loop over each thickness
        for ind in range(len(self.thickness)):
            test_thickness = self.thickness.copy()
            # uniform distribution
            if proposal_distribution == "uniform":
                test_thickness[ind] = param_bounds["thickness"][
                    0
                ] + np.random.uniform() * (
                    param_bounds["thickness"][1] - param_bounds["thickness"][0]
                )
            elif proposal_distribution == "cauchy":
                # cauchy distribution
                test_thickness[ind] += (
                    self.sigma_model["thickness"]
                    * thickness_scale
                    * np.tan(np.pi * (np.random.uniform() - 0.5))
                )

            # assemble test params and check bounds
            test_model, valid_params = self.get_velocity_model(
                param_bounds, test_thickness, self.vel_s.copy()
            )

            # check acceptance criteria
            acc = self.acceptance_criteria(
                test_model, valid_params, data, sample_prior=sample_prior, T=T
            )
            if acc:
                self.thickness = test_thickness.copy()
                self.swap_acc += 1
            elif not sample_prior:
                self.swap_prop += 1

        for ind in range(len(self.vel_s)):
            test_vel_s = self.vel_s.copy()
            if proposal_distribution == "uniform":
                # uniform distribution
                test_vel_s[ind] = param_bounds["vel_s"][0] + np.random.uniform() * (
                    param_bounds["vel_s"][1] - param_bounds["vel_s"][0]
                )
            elif proposal_distribution == "cauchy":
                # cauchy distribution
                test_vel_s[ind] += (
                    self.sigma_model["vel_s"]
                    * vel_s_scale
                    * np.tan(np.pi * (np.random.uniform() - 0.5))
                )

            test_model, valid_params = self.get_velocity_model(
                param_bounds, self.thickness.copy(), test_vel_s
            )

            # check acceptance criteria
            acc = self.acceptance_criteria(
                test_model, valid_params, data, sample_prior=sample_prior, T=T
            )
            if acc:
                self.vel_s = test_vel_s.copy()
                self.swap_acc += 1
            elif not sample_prior:
                # self.vel_s = self.vel_s.copy()
                self.swap_prop += 1

    def acceptance_criteria(self, test_model, valid_params, data, T, sample_prior):
        if not valid_params:
            return False

        if sample_prior:
            logL_new, data_pred_new = 1, np.empty(data.n_data)
            # return True
        else:
            try:
                logL_new, data_pred_new = self.get_likelihood(test_model, data)
            except (DispersionError, ZeroDivisionError):
                # add specific condition here for failed forward model
                self.swap_err += 1
                return False

        # Compute likelihood ratio in log space:
        dlogL = logL_new - self.logL
        if T is not None:
            dlogL = dlogL / T

        xi = np.random.uniform(1)
        # Apply MH criterion (accept/reject)
        if xi <= np.exp(-dlogL):
            self.swap_acc += 1
            self.logL = logL_new
            self.data_pred = data_pred_new
            return True
        else:
            self.swap_prop += 1
            return False

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

            cov_inv = data.data_cov
            cov_inv[cov_inv != 0] = 1 / data.data_cov[cov_inv != 0]

            logL = residuals.T @ cov_inv @ residuals

            return logL, data_pred

        except (DispersionError, ZeroDivisionError) as e:
            raise e

    ### CODE TO ADD IN LATER ###

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
