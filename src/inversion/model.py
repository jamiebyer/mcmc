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
        :param beta: inverse temperature; larger values explore less of the parameter space,
            but are more precise; between 0 and 1
        """
        # initialize

        self.logL = None
        self.data_pred = None

        self.model_params = model_params
        self.sigma_data = sigma_data

        # get generic param_bounds and posterior width from model_params
        self.param_bounds = model_params.assemble_param_bounds()
        self.posterior_width = model_params.assemble_posterior_width()

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

    def perturb_params(
        self,
        data,
        proposal_distribution,
        T=1,
        sample_prior=False,
    ):
        """
        loop over each model parameter, perturb its value, validate the value,
        calculate likelihood, and accept the new model with a probability.

        :param scale_factor:
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

            # cov_inv = data.data_cov
            # cov_inv[cov_inv != 0] = 1 / data.data_cov[cov_inv != 0]
            logL = -np.sum(residuals**2) / (2 * self.sigma_data**2)
            # logL = residuals.T @ cov_inv @ residuals

            return logL, data_pred

        except (DispersionError, ZeroDivisionError) as e:
            raise e
