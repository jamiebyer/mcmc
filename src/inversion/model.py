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
        n_cov_chunk,
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

        if isinstance(self.sigma_data, float):
            self.cov_data_inv = np.diag(
                self.model_params.n_model_params * [1 / self.sigma_data**2]
            )
        else:
            self.cov_data_inv = np.diag(1 / self.sigma_data**2)  # ***

        # get generic param_bounds and posterior width from model_params
        self.param_bounds = model_params.assemble_param_bounds()
        self.proposal_width = model_params.assemble_proposal_width()

        n_params = self.model_params.n_model_params

        # acceptance ratio for each parameter
        # model params will have specific forward model error conditions...
        self.acceptance_rate = {
            # "n_prop": np.zeros(n_params),  # proposed
            "n_acc": 0,  # np.zeros(n_params),  # accepted
            "n_rej": 0,  # np.zeros(n_params),  # rejected
            "n_bounds_err": 0,  # np.zeros(n_params),  # out of bounds error
            "n_physics_err": 0,  # np.zeros(n_params),  # half-space error
            "n_fm_err": 0,  # np.zeros(n_params),  # forward model error
            "acc_rate": 0,  # np.zeros(n_params),
            "bounds_err_ratio": 0,  # np.zeros(n_params),
            "physics_err_ratio": 0,  # np.zeros(n_params),
            "fm_err_ratio": 0,  # np.zeros(n_params),
        }

        self.beta = beta  # beta for parallel tempering

        # params for rotating params
        self.n_cov_chunk = n_cov_chunk
        self.rotation_matrix = np.eye(n_params)
        self.params_sum = np.zeros(n_params)
        self.params_prev = np.zeros(n_params)  # saving the previous accepted params
        self.cov_mat_sum = np.zeros((n_params, n_params))
        self.cov_mat = np.zeros((n_params, n_params))  # model covariance matrix
        self.n_cov = 0  # number of covariance matrices calculated

    def validate_bounds(self, model_params):
        """
        validate bounds.
        """
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
        n_burn,
        n_chunk,
        T=1,
        rotate_params=True,
        sample_prior=False,
        linear_rotation=False,
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

        # if rotate and (np.mod(n_step + 1, n_chunk) == 0):
        if rotate_params:
            burn_in = n_step < n_burn
            # update rotation matrix
            if burn_in and linear_rotation:
                self.rotation_matrix, self.proposal_width = self.linear_rotation(
                    model_params_norm, data
                )
            elif (n_step < (n_burn + self.n_cov_chunk)) & (
                n_step % self.n_cov_chunk == 0
            ):
                # get rotation matrix and step size from correlation matrix
                # lambd, U = np.linalg.eig(self.cov_mat)
                self.rotation_matrix, self.proposal_width = (
                    self.update_covariance_matrix(model_params_norm)
                )

        # *** pick a random parameter to perturb
        ind = np.random.randint(self.model_params.n_model_params)
        # copy current model params to perturb
        #
        if rotate_params:
            test_model_params = self.rotation_matrix.T @ model_params_norm
        else:
            test_model_params = model_params_norm.copy()

        # perturb one param from proposal
        if proposal_distribution == "uniform":
            # uniform distribution
            test_model_params[ind] = self.param_bounds[ind][0] + np.random.uniform()
        elif proposal_distribution == "cauchy":
            # cauchy distribution
            test_model_params[ind] += self.proposal_width[ind] * np.tan(
                np.pi * (np.random.uniform() - 0.5)
            )

        if rotate_params:
            # rotate back
            test_model_params = self.rotation_matrix @ test_model_params

        # reverse normalization (to check forward model)
        # could be before or after checking bounds...
        test_model_params = (
            test_model_params * self.param_bounds[:, 2]
        ) + self.param_bounds[:, 0]

        # sort layers
        test_model_params = self.model_params.sort_layers(test_model_params)

        acc, valid_params = False, True
        if sample_prior:
            # for testing and sampling the prior, return perfect likelihood and empty data.
            logL_new, data_pred_new, test_model_params = 1, np.empty(data.n_data)
        else:
            # validate params
            if not self.validate_bounds(
                test_model_params
            ):  # check parameters are in bounds
                self.acceptance_rate["n_bounds_err"] += 1
                valid_params = False
            elif not self.model_params.validate_physics(
                test_model_params
            ):  # validate physics
                self.acceptance_rate["n_physics_err"] += 1
                valid_params = False
            else:
                # run forward model to predict data for test_params
                try:
                    data_pred_new = self.model_params.forward_model(
                        data.periods, test_model_params
                    )
                except (DispersionError, ZeroDivisionError, TypeError):
                    self.acceptance_rate["n_fm_err"] += 1
                    valid_params = False

        if valid_params:
            # calculate likelihood with predicted data
            logL_new = Model.get_likelihood(data, data_pred_new, self.sigma_data)
            # check acceptance criteria
            acc = self.acceptance_criteria(logL_new, T=T)
            if acc:
                self.params_prev = (
                    self.model_params.model_params - self.param_bounds[:, 0]
                ) / self.param_bounds[:, 2]

                self.model_params.model_params = test_model_params.copy()
                self.logL = logL_new
                self.data_pred = data_pred_new

        self.update_acceptance_rate(acc)
        # self.stepsize_tuning(n_step)

    #
    # PARAM ROTATIONS
    #

    def update_covariance_matrix(self, model_params_norm):
        """
        track sample mean
        add newest sample to current cov mat
        store sum without averaging
        """

        # take difference between current params and previous params
        params_diff = model_params_norm - self.params_prev

        if np.all(params_diff == 0):
            return self.rotation_matrix, self.proposal_width

        # for each combination of params
        # add to the sum
        self.cov_mat_sum += np.outer(np.transpose(params_diff), params_diff)

        # divide cov mat by number of samples
        self.cov_mat = self.cov_mat_sum / (self.n_cov + 1)
        self.n_cov += 1

        U, s, _ = np.linalg.svd(
            self.cov_mat
        )  # rotate it to its Singular Value Decomposition
        # ut = np.transpose(u)
        lambd = np.sqrt(s)  # pcsd

        return U, lambd

    def linear_rotation(self, model_params, data, var=12):
        """
        during burn-in, when there aren't enough samples to create a reliable covariance matrix,
        use a linear approximation to perform rotation.

        :param var: from trial and error... ***
        """
        jac_mat = self.get_jacobian(model_params, data)

        m_prior_inv = np.diag(
            self.model_params.n_model_params * [var]
        )  # to stabilize matrix inversion
        cov_tmp = jac_mat.T @ self.cov_data_inv @ jac_mat + m_prior_inv

        V, L, _ = np.linalg.svd(cov_tmp)
        pcsd = 0.5 * (1.0 / np.sqrt(np.abs(L)))  # PC standard deviations

        return V, pcsd

    def get_jacobian(self, model_params, data, n_dm=50):
        """
        get jacobian matrix.
        get derivatives for n_dm spacings, and keep the most stable values.

        :param n_dm: number of derivative spacings to try.
        """
        # only should need to create the dm array once.
        # defining dm spacing
        # multiply with params to expand
        dm = np.outer(model_params * 0.1, np.logspace(0, n_dm, base=(1 / 1.5)))
        # define array to hold derivatives. n_dm derivatives for each
        # model parameter, and each data point.
        derivatives = np.zeros((self.model_params.n_model_params, data.n_data, n_dm))

        # estimate derivatives for range of dm values
        for dm_ind in range(n_dm):
            for param_ind in range(self.model_params.n_model_params):
                model_pos = model_params.copy()
                model_neg = model_params.copy()

                model_pos[param_ind] = model_pos[param_ind] + dm[param_ind, dm_ind]
                model_neg[param_ind] = model_neg[param_ind] - dm[param_ind, dm_ind]

                # *** bc of the depth swapping, forward_model rn returns updated model params too
                # get predicted data for pos and neg model params
                data_pos, _ = self.model_params.forward_model(data.periods, model_pos)
                data_neg, _ = self.model_params.forward_model(data.periods, model_neg)

                # check that data_pos and data_neg aren't too far apart...
                # *** set to zero to avoid numerical instability
                inds = np.abs((data_pos - data_neg) / (data_pos + data_neg)) > 1.0e-7
                # centered derivative
                derivatives[param_ind, inds, dm_ind] = (
                    data_pos[inds] - data_neg[inds]
                ) / (2.0 * dm[param_ind, dm_ind])

        jac_mat = self.get_best_derivative(derivatives, data, n_dm)

        jac_mat = (
            jac_mat * self.param_bounds[:, 2]
        )  # scale columns of Jacobian for stability
        return jac_mat

    def get_best_derivative(self, derivatives, data, n_dm, min_bound=1.0e-7):
        """
        finding a stable value for the derivative for each parameter.
        looking for a derivative that doesn't change much for a few different spacings.
        populate Jacobian matrix with stable derivatives.

        :param derivatives:
        :param n_dm:
        :param min_bound:
        """
        jac_mat = np.zeros((data.n_data, self.model_params.n_model_params))
        variation = np.zeros((self.model_params.n_model_params, data.n_data, n_dm - 2))

        # derivatives smaller than min_bound are set to 0
        small_inds = derivatives < min_bound
        derivatives[small_inds] = 0

        inds = np.logical_not(small_inds)
        good_inds = np.empty(
            (self.model_params.n_model_params, data.n_data, n_dm - 2), dtype=bool
        )

        for m_ind in range(good_inds.shape[0]):
            for d_ind in range(good_inds.shape[1]):
                for i in range(1, inds.shape[2] - 1):
                    # good inds have a good ind on both sides
                    good_inds[m_ind, d_ind, i - 1] = inds[m_ind, d_ind, i : i + 3].all()

        # good_inds = np.array(
        #    [i for i in range(1, inds.shape[2] - 1) if inds[:, :, i : i + 3].all()]
        # )

        # for dm_ind in np.arange(1, n_dm - 1):
        # find derivatives which have the least variation
        variation[good_inds] = np.abs(
            (
                (derivatives[:, :, :-2][good_inds] / derivatives[:, :, 1:-1][good_inds])
                + (
                    derivatives[:, :, 1:-1][good_inds]
                    / derivatives[:, :, 2:][good_inds]
                )
            )
            / 2
            - 1
        )

        for d_ind in range(jac_mat.shape[0]):
            for m_ind in range(jac_mat.shape[1]):
                # check that there is a good ind in this col
                valid_inds = good_inds[m_ind, d_ind]
                if np.any(valid_inds):
                    # find the min
                    min_ind = np.argmin(variation[m_ind, d_ind, valid_inds])
                    # set jac value
                    jac_mat[d_ind, m_ind] = derivatives[m_ind, d_ind, min_ind + 1]

        return jac_mat

    #
    # ACCEPTANCE CRITERIA
    #

    @staticmethod
    def get_likelihood(data, data_pred, sigma_data):
        """
        :param velocity_model:
        :param data:
        """
        residuals = data.data_obs - data_pred

        # cov_inv = data.data_cov
        # cov_inv[cov_inv != 0] = 1 / data.data_cov[cov_inv != 0]
        # logL = residuals.T @ cov_inv @ residuals

        # for identical errors
        # logL = -np.sum(residuals**2) / (2 * self.sigma_data**2)
        logL = -np.sum((residuals**2) / (2 * sigma_data**2))

        return logL

    def acceptance_criteria(self, logL_new, T):
        """
        determine whether to accept the test model parameters.
        for sampling the prior, always accept the new parameters.

        Compute likelihood. Use metropolis-hastings
        acceptance criteria to accept / reject based on model likelihood.

        :param T:
        """

        # Compute likelihood ratio in log space
        # T=1 by default, unless performing optimization inversion
        dlogL = logL_new - self.logL
        dlogL = dlogL / T

        # Apply Metropolis-Hastings criterion (accept/reject)
        xi = np.random.uniform()  # between 0 and 1
        return xi <= np.exp(dlogL)

    def update_acceptance_rate(self, acc):
        """
        update acceptance rate.
        make sure no division by zero.

        :param acc: boolean for whether perturbed parameter is accepted.
        :paran ind: index of parameter to update.
        """
        if acc:
            self.acceptance_rate["n_acc"] += 1
        else:
            self.acceptance_rate["n_rej"] += 1

        if self.acceptance_rate["n_rej"] > 0:
            self.acceptance_rate["acc_rate"] = self.acceptance_rate["n_acc"] / (
                self.acceptance_rate["n_acc"] + self.acceptance_rate["n_rej"]
            )
            self.acceptance_rate["bounds_err_ratio"] = self.acceptance_rate[
                "n_bounds_err"
            ] / (self.acceptance_rate["n_acc"] + self.acceptance_rate["n_rej"])
            self.acceptance_rate["physics_err_ratio"] = self.acceptance_rate[
                "n_physics_err"
            ] / (self.acceptance_rate["n_acc"] + self.acceptance_rate["n_rej"])
            self.acceptance_rate["fm_err_ratio"] = self.acceptance_rate["n_fm_err"] / (
                self.acceptance_rate["n_acc"] + self.acceptance_rate["n_rej"]
            )

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
        # if the acceptance rate is too high, increase the step size
        # if the acceptance rate is too low, decrease the step size
        # adapt the acceptance rate less and less as more steps are taken
        # diff = np.abs(acc_rate - acc_optimal)
        high_inds = self.acceptance_rate["acc_rate"] > 0.4
        low_inds = (self.acceptance_rate["acc_rate"] > 0) & (
            self.acceptance_rate["acc_rate"] < 0.2
        )
        self.proposal_width[high_inds] = (
            self.proposal_width[high_inds] * 1.5
        )  # gamma*diff
        self.proposal_width[low_inds] = self.proposal_width[low_inds] * 0.5


class SyntheticModel(Model):
    pass


class GeneratedModel(Model):

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
