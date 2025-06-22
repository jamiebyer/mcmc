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

        self.model_params = model_params

        # get generic param_bounds and sigma from model_params
        self.param_bounds = model_params.assemble_param_bounds()
        self.sigma_model = model_params.assemble_sigma_model()

        # acceptance ratio for each parameter
        self.swap_acc = np.zeros(self.model_params.n_model_params)
        self.swap_rej = np.zeros(self.model_params.n_model_params)
        self.swap_err = np.zeros(self.model_params.n_model_params)

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

    def validate_bounds(self, model_params, ind):
        """
        validate bounds.
        """
        # technically only need to validate one changed ind
        valid_params = np.all(
            all(model_params >= self.param_bounds[:, 0])
            & all(model_params <= self.param_bounds[:, 1])
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

    def stepsize_tuning():
        """
        Step size can be found by trial and error.
        This is sometimes done during burn in, where we target an acceptance rate of ~30%.
        When a << 30%, decrease step size; when a>>30%, increase step size.
        Once an acceptable step size it found, stop adapting.
        Sometimes, we do this by way of diminishing adaptation (see Rosenthal in Handbook of MCMC).
        Adapt the step size less and less as the sampler progresses, until adaptation vanishes.
        """
        pass

    def rotate_params():
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

    def perturb_params(
        self,
        data,
        proposal_distribution,
        rotation=False,
        T=1,
        sample_prior=False,
    ):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param scale_factor:
        """
        # can normalize regardless
        if rotation:
            pass

        # randomly select a model parameter to perturb
        ind = np.random.randint(self.model_params.n_model_params)

        # copy current model params to perturb
        test_model_params = self.model_params.model_params.copy()

        # uniform distribution
        if proposal_distribution == "uniform":
            test_model_params[ind] = (
                self.param_bounds[ind][0]
                + np.random.uniform() * self.param_bounds[ind][2]
            )
        elif proposal_distribution == "cauchy":
            # cauchy distribution
            test_model_params[ind] += (
                self.sigma_model[ind]
                * self.param_bounds[ind][2]
                * np.tan(np.pi * (np.random.uniform() - 0.5))
            )

        # assemble test params and check bounds
        test_model = self.model_params.get_velocity_model(test_model_params)

        # check bounds
        valid_params = self.validate_bounds(test_model_params, ind)

        if valid_params:
            # check acceptance criteria
            acc = self.acceptance_criteria(
                test_model, ind, data, sample_prior=sample_prior, T=T
            )
            if acc:
                self.model_params.model_params = test_model_params.copy()
                self.swap_acc[ind] += 1
            else:
                self.swap_rej[ind] += 1
        else:
            self.swap_rej[ind] += 1

    def acceptance_criteria(self, test_model, ind, data, T, sample_prior):
        if sample_prior:
            logL_new, data_pred_new = 1, np.empty(data.n_data)
        else:
            try:
                logL_new, data_pred_new = self.get_likelihood(test_model, data)
            except (DispersionError, ZeroDivisionError):
                # add specific condition here for failed forward model
                self.swap_err[ind] += 1
                return False

        # Compute likelihood ratio in log space
        # T=1 by default, unless performing optimization inversion
        dlogL = logL_new - self.logL
        dlogL = dlogL / T

        # Apply MH criterion (accept/reject)
        xi = np.random.uniform()  # between 0 and 1
        if xi <= np.exp(dlogL):
            self.logL = logL_new
            self.data_pred = data_pred_new
            return True
        else:
            return False

    def get_likelihood(self, velocity_model, data):
        """
        :param model_params: params to calculate likelihood with
        """
        # return 1, periods
        try:
            data_pred = self.model_params.forward_model(data.periods, velocity_model)
            residuals = data.data_obs - data_pred
            # for identical errors
            # logL = np.sum(residuals**2) / (self.sigma_model**2)

            cov_inv = data.data_cov
            cov_inv[cov_inv != 0] = 1 / data.data_cov[cov_inv != 0]
            logL = -np.sum(residuals**2) / (2 * 0.1**2)
            # logL = residuals.T @ cov_inv @ residuals

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
