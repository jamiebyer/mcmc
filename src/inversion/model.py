import numpy as np
from disba._exception import DispersionError
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
        T=1,
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
        # can normalize

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
                self.posterior_width[ind]
                * self.param_bounds[ind][2]
                * np.tan(np.pi * (np.random.uniform() - 0.5))
            )

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
        self.stepsize_tuning(n_step)

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
            # check that halfspace is the fastest layer
            # check specific criteria for params
            self.model_params.validate_physics()
            self.acceptance_rate["n_hs_err"][ind] += 1
        else:
            try:
                logL_new, data_pred_new, model_params = self.get_likelihood(
                    test_model_params, data
                )
            except (DispersionError, ZeroDivisionError):
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

        except (DispersionError, ZeroDivisionError) as e:
            raise e

    def stepsize_tuning(self, n_step):
        """
        Step size can be found by trial and error.
        This is sometimes done during burn in, where we target an acceptance rate of ~30%.
        When a << 30%, decrease step size; when a >> 30%, increase step size.
        Once an acceptable step size it found, stop adapting.
        Sometimes, we do this by way of diminishing adaptation (see Rosenthal in Handbook of MCMC).
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
        low_inds = self.acceptance_rate["acc_rate"] < 0.2
        self.posterior_width[high_inds] = (
            self.posterior_width[high_inds] * 0.9
        )  # gamma*diff
        self.posterior_width[low_inds] = self.posterior_width[low_inds] * 1.1

    def update_covariance_matrix(self, n_step):
        # need prev matrix to compute current.
        # subtract prev mean cov and current mean cov mat.

        # update mean params
        self.mean_params = (1 / n_step) * (
            ((n_step - 1) / n_step) * (self.mean_params)
            + self.model_params.model_params
        )
        self.cov_mat = (1 / n_step) * (
            np.sum(self.model_params.model_params - self.mean_params) ** 2
        )
