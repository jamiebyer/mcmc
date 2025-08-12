import numpy as np
from disba._exception import DispersionError
from numba.core.errors import TypingError
import pandas as pd

np.complex_ = np.complex64


class Model:

    def __init__(
        self,
        model_params,
        sigma_data,
        beta=None,
    ):
        """
        :param model_params:
        :param sigma_data:
        :param beta:
        """
        self.logL = None
        self.data_pred = None

        self.model_params = model_params
        self.sigma_data = sigma_data

        # get generic param_bounds and posterior width from model_params
        self.param_bounds = model_params.assemble_param_bounds()
        self.posterior_width = model_params.assemble_posterior_width()

        n_params = self.model_params.n_model_params

        # acceptance ratio for each parameter
        # model params will have specific forward model error conditions...
        self.acceptance_rate = {
            # "n_prop": np.zeros(n_params),  # proposed
            "n_acc": np.zeros(n_params),  # accepted
            "n_rej": np.zeros(n_params),  # rejected
            "n_fm_err": np.zeros(n_params),  # forward model error
            "n_hs_err": np.zeros(n_params),  # half-space error
            "acc_rate": np.zeros(n_params),
            "fm_err_ratio": np.zeros(n_params),
            "fm_hs_ratio": np.zeros(n_params),
        }

        self.beta = beta

        self.params_sum = np.zeros(n_params)
        self.cov_mat_sum = np.zeros((n_params, n_params))
        self.cov_mat = np.zeros((n_params, n_params))

        # variables for storing and computing covariance matrix after burn-in
        # self.rot_mat = np.eye(self.n_params)  # initialize rotation matrix
        # self.n_cov = 0  # initialize the dividing number for covariance

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

    def generate_model_params(self):
        """
        generating initial params for new model.
        uniform between bounds
        """
        model_params = np.random.uniform(
            self.param_bounds[:, 0],
            self.param_bounds[:, 1],
            self.model_params.n_model_params,
        )

        return model_params

    def perturb_params(
        self,
        data,
        proposal_distribution,
        n_step,
        burn_in,
        T=1,
        rotate=True,
        sample_prior=False,
    ):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param data:
        :param proposal_distribution:
        :param T:
        :param sample_prior:
        """
        # normalize (by parameter bounds...)
        model_params_norm = (
            self.model_params.model_params - self.param_bounds[:, 0]
        ) / self.param_bounds[:, 2]

        if rotate:
            model_params_rot, U, self.posterior_width = self.rotate_params(
                model_params_norm
            )

        # perturb each parameter individually
        # ***
        for ind in range(self.model_params.n_model_params):
            # copy current model params to perturb
            #
            if rotate:
                test_model_params = model_params_rot
            else:
                test_model_params = self.model_params.model_params.copy()

            if proposal_distribution == "uniform":
                # uniform distribution
                test_model_params[ind] = self.param_bounds[ind][0] + np.random.uniform()
            elif proposal_distribution == "cauchy":
                # cauchy distribution
                test_model_params[ind] += self.posterior_width[ind] * np.tan(
                    np.pi * (np.random.uniform() - 0.5)
                )

        if rotate:
            # rotate back
            test_model_params = U @ test_model_params

        # reverse normalization (to check forward model)
        # could be before or after checking bounds...
        test_model_params = (
            test_model_params * self.param_bounds[:, 2]
        ) + self.param_bounds[:, 0]

        # check bounds
        valid_params = self.validate_bounds(test_model_params, ind)

        acc = False
        if valid_params:
            # assemble test params and check bounds

            # run forward model

            # check acceptance criteria
            acc, test_model_params = self.acceptance_criteria(
                test_model_params, ind, data, sample_prior=sample_prior, T=T
            )

            if acc:
                self.model_params.model_params = test_model_params.copy()

        self.update_acceptance_rate(acc, ind)
        # self.stepsize_tuning(n_step)

    #
    # PARAM ROTATIONS
    #

    def rotate_params(self, model_params, burn_in):
        """
        before  burn-in use Jacobian
        rotate to PC space (after burn in)
        get posterior width / scale for propsal distribution
        get rotation matrix and step size / standard deviation
        """

        # check if it's burn in
        # if it's before burning, do a linear rotation
        if burn_in:
            model_params_rot, U, lamdb, self.linear_rotation()
        else:
            # get rotation matrix and step size from correlation matrix
            lambd, U = np.linalg.eig(self.cov_mat)
            model_params_rot = U.T @ model_params

        return model_params_rot, U, lambd

    def linear_rotation(self, var=12):
        """
        during burn-in, when there aren't enough samples to create a reliable covariance matrix,
        use a linear approximation to perform rotation.

        :param var: from trial and error... ***
        """
        jac_mat = self.get_jacobian()

        m_prior_inv = np.diag(self.n_params * [var])  # to stabilize matrix inversion
        cov_data_inv = np.diag(1 / self.sigma_data**2)  # ***
        cov_tmp = jac_mat.T @ cov_data_inv @ jac_mat + m_prior_inv

        V, L, VT = np.linalg.svd(cov_tmp)
        pcsd = 0.5 * (1.0 / np.sqrt(np.abs(L)))  # PC standard deviations

        return pcsd

    def get_jacobian(self, model_params, n_dm=50):
        """
        get jacobian matrix.
        get derivatives for n_dm spacings, and keep the most stable values.

        :param n_dm: number of derivative spacings to try.
        """
        # defining dm spacing
        dm = self.params * 0.1 * (1 / 1.5) ** np.arange(n_dm)
        # define array to hold derivatives. n_dm derivatives for each
        # model parameter, and each data point.
        derivatives = np.zeros((self.n_params, self.data.n_data, n_dm))

        # estimate derivatives for range of dm values
        for dm_ind in range(n_dm):
            for param_ind in range(self.n_params):
                model_pos = model_params.copy()
                model_neg = model_params.copy()

                model_pos[param_ind] = model_pos[param_ind] + dm[param_ind, dm_ind]
                model_neg[param_ind] = model_neg[param_ind] - dm[param_ind, dm_ind]

                # *** bc of the depth swapping, forward_model rn returns updated model params too
                # get predicted data for pos and neg model params
                data_pos = self.model_params.forward_model(self.data.periods, model_pos)
                data_neg = self.model_params.forward_model(self.data.periods, model_neg)

                # check that data_pos and data_neg aren't too far apart...
                term = np.abs((data_pos - data_neg) / (data_pos + data_neg))
                if term > 1.0e-7:
                    # centered derivative
                    derivatives[param_ind, :, dm_ind] = (data_pos - data_neg) / (
                        2.0 * dm[param_ind, dm_ind]
                    )
                else:
                    # set to zero to avoid numerical instability
                    pass

        jac_mat = self.get_best_derivative(derivatives, self.data.n_data, n_dm)

        jac_mat = (
            jac_mat * self.param_bounds[:, 2]
        )  # scale columns of Jacobian for stability
        return jac_mat

    def get_best_derivative(self, derivatives, n_dm, min_bound=1.0e-7):
        """
        finding a stable value for the derivative for each parameter.
        looking for a derivative that doesn't change much for a few different spacings.
        populate Jacobian matrix with stable derivatives.

        :param derivatives:
        :param n_dm:
        :param min_bound:
        """
        jac_mat = np.zeros((self.data.n_data, self.n_params))
        variation = np.empty(self.n_params, self.data.n_data, n_dm - 2)

        # set derivatives smaller than min_bound to NaN (and get indices)
        nan_inds = derivatives < min_bound

        for dm_ind in range(n_dm - 2):
            # find derivatives which have the least variation
            variation[:, :, dm_ind] = np.abs(
                np.sum(
                    (
                        derivatives[not nan_inds][:, :, dm_ind : dm_ind + 1]
                        / derivatives[not nan_inds][:, :, dm_ind + 1 : dm_ind + 2]
                    )
                    / 2
                    - 1,
                    axis=2,
                )
            )
        # find min indices
        min_inds = np.nanmin(variation, axis=2)  # plus 1 ***
        # set Jacobian values
        jac_mat = derivatives[min_inds]

        # if ind is nan, set Jac ind to 0

        return jac_mat

    #
    # ACCEPTANCE CRITERIA
    #

    def acceptance_criteria(self, test_model_params, ind, data, T, sample_prior):
        """
        determine whether to accept the test model parameters.
        for sampling the prior, always accept the new parameters.

        Run the forward model with the test parameters to validate parameters, and
        to get predicted data from this model. Compute likelihood. Use metropolis-hastings
        acceptance criteria to accept / reject based on model likelihood.

        :param test_model_params:
        :param ind:
        :param data:
        :param T:
        :param sample_prior:
        """
        if sample_prior:
            # for testing and sampling the prior, return perfect likelihood and empty data.
            logL_new, data_pred_new, test_model_params = 1, np.empty(data.n_data)
        elif not self.model_params.validate_physics():
            # *** separate function for validating params and running forward model?
            # check that halfspace is the fastest layer
            # check specific criteria for params
            self.model_params.validate_physics()
            self.acceptance_rate["n_hs_err"][ind] += 1
        else:
            try:
                logL_new, data_pred_new, model_params = self.get_likelihood(
                    test_model_params, data
                )
            except (DispersionError, ZeroDivisionError, TypingError):
                # add specific condition here for failed forward model
                self.acceptance_rate["n_fm_err"][ind] += 1
                return False, None

        # Compute likelihood ratio in log space
        # T=1 by default, unless performing optimization inversion
        dlogL = logL_new - self.logL
        dlogL = dlogL / T

        # Apply Metropolis-Hastings criterion (accept/reject)
        xi = np.random.uniform()  # between 0 and 1
        if xi <= np.exp(dlogL):
            self.logL = logL_new
            self.data_pred = data_pred_new
            return True, model_params
        else:
            return False, model_params

    def update_acceptance_rate(self, acc, ind):
        """
        update acceptance rate.
        make sure no division by zero.

        :param acc: boolean for whether perturbed parameter is accepted.
        :paran ind: index of parameter to update.
        """
        if acc:
            self.acceptance_rate["n_acc"][ind] += 1
        else:
            self.acceptance_rate["n_rej"][ind] += 1

        if self.acceptance_rate["n_rej"][ind] > 0:
            self.acceptance_rate["acc_rate"][ind] = self.acceptance_rate["n_acc"][
                ind
            ] / (
                self.acceptance_rate["n_acc"][ind] + self.acceptance_rate["n_rej"][ind]
            )

    def get_likelihood(self, model_params, data):
        """
        :param velocity_model:
        :param data:
        """
        try:
            data_pred, model_params = self.model_params.forward_model(
                data.periods, model_params
            )
            residuals = data.data_obs - data_pred

            # cov_inv = data.data_cov
            # cov_inv[cov_inv != 0] = 1 / data.data_cov[cov_inv != 0]
            # logL = residuals.T @ cov_inv @ residuals

            # for identical errors
            logL = -np.sum(residuals**2) / (2 * self.sigma_data**2)

            return logL, data_pred, model_params

        except (DispersionError, ZeroDivisionError, TypeError) as e:
            raise e

    #
    # STEPSIZE TUNING
    #

    def stepsize_tuning(self, n_step):
        """
        Step size can be found by trial and error.
        This is sometimes done during burn in, where we target an acceptance rate of ~30%.
        When a << 30%, decrease step size; when a >> 30%, increase step size.
        Adapt the step size less and less as the sampler progresses, until adaptation vanishes.
        """

        # tune parameter scales using acceptance rate
        # update scale
        acc_optimal = 0.3
        # acc_optimal = 0.234

        # gamma = np.linspace(1, 0, 10000)
        gamma = 1 / (n_step + 1)
        """
        self.posterior_width = np.exp(
            np.log(self.posterior_width) + gamma * (acc_rate - acc_optimal)
        )
        """
        # if the acceptance rate is too high, increase the step size
        # if the acceptance rate is too low, decrease the step size
        # adapt the acceptance rate less and less as more steps are taken
        # diff = np.abs(acc_rate - acc_optimal)
        high_inds = self.acceptance_rate["acc_rate"] > 0.4
        low_inds = (self.acceptance_rate["acc_rate"] > 0) & (
            self.acceptance_rate["acc_rate"] < 0.2
        )
        self.posterior_width[high_inds] = (
            self.posterior_width[high_inds] * 1.5
        )  # gamma*diff
        self.posterior_width[low_inds] = self.posterior_width[low_inds] * 0.5

    def update_covariance_matrix(self, n_step):
        # track sample mean
        # add newest sample to current cov mat
        # store sum without averaging

        # update mean params
        self.params_sum += self.model_params.model_params
        mean_params = self.params_sum / n_step

        # for each combination of params
        # add to the sum
        self.cov_mat_sum += (self.model_params.model_params - mean_params) @ (
            self.model_params.model_params - mean_params
        ).T

        # divide cov mat by number of samples
        self.cov_mat = self.cov_mat_sum / n_step
