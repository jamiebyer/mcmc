import numpy as np
import sys

from disba import PhaseDispersion
from disba._exception import DispersionError
from inversion.model import Model
from dask.distributed import Client
import os
import xarray as xr
import time
from copy import deepcopy
import dask.dataframe as dd
import time


class Inversion:
    def __init__(
        self,
        data,
        model_params,
        sigma_data,
        n_burn,
        n_chunk,
        n_mcmc,
        beta_spacing_factor=None,
        n_chains=1,
        out_filename=None,
    ):
        """
        :param n_data: number of data observed.

        :param n_chains: number of chains
        :param beta_spacing_factor:

        :param n_bins: number of bins for the model histograms
        :param n_burn_in: number of steps to discard from the start of the run (to avoid bias towards the starting model)
        :param n_chunk: number of steps/iterations to save to file at a time. determines total number of steps.
        :param n_rot: number of steps to do after nonlinear rotation starts
        """

        """
        run options:
        - run with optimized starting model
        - run with burn-in
        - run with linearlization
        - run with parallel tempering/ chains
        """

        self.data = data

        # parameters related to number of steps taken in random walk
        self.n_burn = n_burn
        self.n_chunk = n_chunk
        self.n_mcmc = n_mcmc  # number of steps for the random walk

        # initialize chains, generate starting params.
        self.model_params = model_params
        self.n_chains = n_chains
        self.initialize_chains(
            model_params,
            sigma_data,
            beta_spacing_factor,
        )

        # set initial likelihood ***

        # *** need a check for if the file exists ***
        # define out_dir
        self.out_dir = "./results/inversion/"
        if not out_filename:
            self.out_filename = str(int(time.time()))
        else:
            self.out_filename = out_filename

        # save input inversion params and data info
        # create dataset
        # update with dictionary from data class
        self.define_input_dataset(data, model_params)

    def define_input_dataset(self, data, model_params):
        # get data dict and model params dict
        input_dict = {"coords": {}, "data_vars": {}, "attrs": {}}

        m_dict = model_params.get_model_params_dict()
        d_dict = data.get_data_dict()

        input_dict["coords"].update(m_dict["coords"])
        input_dict["coords"].update(d_dict["coords"])

        input_dict["data_vars"].update(m_dict["data_vars"])
        input_dict["data_vars"].update(d_dict["data_vars"])

        input_dict["attrs"].update(m_dict["attrs"])
        input_dict["attrs"].update(d_dict["attrs"])

        input_dict["attrs"]["n_burn"] = self.n_burn

        input_dict["dims"] = {
            k: len(v["data"]) for k, v in input_dict["coords"].items()
        }

        input_ds = xr.Dataset.from_dict(input_dict)

        # if the output folder doesn't exist, create it.
        out_path = self.out_dir + "input-" + self.out_filename + ".nc"
        # saved dataset should just have parameter names, dimensions, and constants
        if not os.path.isfile(out_path):
            input_ds.to_netcdf(out_path)

    def define_results_dataset(self, model_params):
        # write empty dataset to file, with coord for n_data
        coords = {
            # "periods": self.data.periods,
            "step": np.array([]),
        }
        ds = xr.Dataset(coords=coords)

        out_path = self.out_dir + "results-" + self.out_filename + ".nc"
        if not os.path.isfile(out_path):
            ds.to_netcdf(out_path)

        self.ds_storage = {
            "coords": {
                "period": {"dims": "period", "data": self.data.periods},
                "n_model_params": {
                    "dims": "n_model_params",
                    "data": np.arange(model_params.n_model_params),
                },
                "n_model_params_y": {
                    "dims": "n_model_params_y",
                    "data": np.arange(model_params.n_model_params),
                },
                "step": {"dims": "step", "data": np.arange(self.n_chunk)},
            },
            "data_vars": {
                "model_params": {
                    "dims": ["n_model_params", "step"],
                    "data": np.empty((model_params.n_model_params, self.n_chunk)),
                },
                "logL": {"dims": ["step"], "data": np.empty(self.n_chunk)},
                "cov_mat": {
                    "dims": ["n_model_params", "n_model_params_y", "step"],
                    "data": np.empty(
                        (
                            model_params.n_model_params,
                            model_params.n_model_params,
                            self.n_chunk,
                        )
                    ),
                },
                "acc_rate": {
                    "dims": ["n_model_params", "step"],
                    "data": np.empty((model_params.n_model_params, self.n_chunk)),
                },
                "fm_err": {
                    "dims": ["n_model_params", "step"],
                    "data": np.empty((model_params.n_model_params, self.n_chunk)),
                },
                "hs_err": {
                    "dims": ["n_model_params", "step"],
                    "data": np.empty((model_params.n_model_params, self.n_chunk)),
                },
                "data_pred": {
                    "dims": ["period", "step"],
                    "data": np.empty((self.data.n_data, self.n_chunk)),
                },
            },
        }

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

        # need to add temp to saving dataset.

        # self.define_input_dataset()
        self.define_results_dataset()

        # reduce logarithmically
        for T in temps:
            print("\ntemp", T)
            # number of steps at this temperature
            for _ in range(n_steps):
                print(_)
                # perturb each parameter
                # if using the perturb params function, it does the acceptance rate stats in the function
                self.perturb_params(param_bounds, data, T=T)

                self.store_samples()

        self.write_samples()

    def get_betas(self, beta_spacing_factor):
        """
        getting beta values to use for each chain.

        :param beta_spacing_factor: determines the spacing between values of beta.

        """
        # smaller spacing will have higher acceptance rates. larger spacing will explore more of the space.
        # we want to tune the spacing so our acceptance rate is 30-50%. dTlog should be larger than 1.
        # change beta steps vals based on acceptance rate

        # *** later could tune beta_spacing_factor as the inversion runs, looking at the acceptance rate every ~10 000 steps ***
        # :param beta: inverse temperature; larger values explore less of the parameter space,
        #    but are more precise; between 0 and 1

        if self.n_chains == 1:
            return [1]

        # getting beta values to use for parallel tempering
        n_temps = self.n_chains
        # 1/4 to 1/2 of the chains should be beta=1.
        n_temps_frac = int(np.ceil(n_temps / 4))
        betas = np.zeros(n_temps, dtype=float)
        inds = np.arange(n_temps_frac, n_temps)
        betas[inds] = 1.0 / beta_spacing_factor**inds

        return betas

    def initialize_chains(
        self,
        model_params,
        sigma_data,
        beta_spacing_factor,
        # optimize_starting_model=False,
    ):
        """
        initialize each of the chains, setting starting parameters, beta values, initial rotation params

        :param model_params:
        :param sigma_data:
        :param beta_spacing_factor
        """

        # generate the starting models
        betas = self.get_betas(beta_spacing_factor)  # get values of beta
        chains = []
        for ind in range(self.n_chains):
            model = Model(
                deepcopy(model_params),
                sigma_data,
                betas[ind],
            )

            # initialize model params
            valid_params = False
            while not valid_params:
                # generate model params between bounds.
                test_params = model.generate_model_params()

                # get likelihood
                try:
                    logL, data_pred, test_params = model.get_likelihood(
                        test_params, self.data
                    )
                    valid_params = True
                except (DispersionError, ZeroDivisionError) as _:
                    valid_params = False

            model.model_params.model_params = test_params
            model.logL = logL
            model.data_pred = data_pred

            chains.append(model)

        self.chains = chains

    def random_walk(
        self,
        model_params,
        proposal_distribution,
        sample_prior=False,
    ):
        """
        perform the main loop, for n_mcmc iterations.
        """
        start_time = time.time()
        out_path = self.out_dir + "results-" + self.out_filename + ".nc"
        # create empty file to save results
        # define dict for saving results
        self.define_results_dataset(model_params)

        # all chains need to be on the same step number to compare
        for n_steps in range(self.n_mcmc):
            # burn_in = n_steps < self.n_burn
            delayed_results = []  # format for parallelizing later
            for ind in range(self.n_chains):
                chain_model = self.chains[ind]
                chain_model.perturb_params(
                    self.data,
                    proposal_distribution,
                    n_steps,
                    sample_prior=sample_prior,
                )

                delayed_results.append(chain_model)

            # synchronizing the separate chains
            self.chains = delayed_results

            if self.n_chains > 1:
                # tempering exchange move
                self.perform_tempering_swap()

            # store every sample; only write to file every n_chunk samples.
            self.store_samples(n_steps)
            # update cov matrix (every n steps)
            if n_steps > self.n_burn:
                chain_model.update_covariance_matrix(n_steps)
            self.write_samples(n_steps, out_path)

        # add most probable model to file
        # (and total computation time)
        self.write_probable_model(out_path, start_time)

    def store_samples(self, n_step):
        """
        write out to .zarr in chunks of size n_keep.

        :param n_step: current step number in the random walk.
        """
        # the step size on ds_storage should be up to n_batch, but should start at n_steps for the written ds.
        n_save = n_step % self.n_chunk

        for chain in self.chains:
            # saving the chain model with beta of 1
            if chain.beta == 1:
                # update storage dataset with new param values
                self.ds_storage["data_vars"]["model_params"]["data"][
                    :, n_save
                ] = chain.model_params.model_params.copy()
                self.ds_storage["data_vars"]["logL"]["data"][n_save] = chain.logL
                self.ds_storage["data_vars"]["cov_mat"]["data"][
                    :, :, n_save
                ] = chain.cov_mat
                self.ds_storage["data_vars"]["data_pred"]["data"][
                    :, n_save
                ] = chain.data_pred.copy()
                # self.ds_storage["beta"][{"step": n_step}] = chain.beta

                # acceptance rate
                self.ds_storage["data_vars"]["acc_rate"]["data"][:, n_save] = (
                    chain.acceptance_rate["acc_rate"]
                )

                """
                # error ratio
                self.ds_storage["data_vars"]["fm_err"]["data"] = chain.acceptance_rate[
                    "n_fm_err"
                ] / (chain.acceptance_rate["n_acc"] + chain.acceptance_rate["n_rej"])
                self.ds_storage["data_vars"]["hs_err"]["data"] = chain.acceptance_rate[
                    "n_hs_err"
                ] / (chain.acceptance_rate["n_acc"] + chain.acceptance_rate["n_rej"])
                """

    def write_samples(self, n_steps, out_path):
        """
        append current samples to file

        currently opening file, concatenating, and closing.
        netcdf cannot append along a dimension currently...
        zarr has an issue with the compression when saving...
        later could save each chunk in an individual file and concat at the end...

        :param n_steps:
        :param out_dir:
        """
        # save n_chunk samples at a time
        # n_samples should be self.n_chunk until convergence, where it could be uneven.
        write_samples = np.mod(n_steps + 1, self.n_chunk) == 0

        if not write_samples:
            return

        percent = ((n_steps + 1) / self.n_mcmc) * 100
        print(str(np.round(percent, 1)) + " %")

        # *** change later to use context manager ***
        ds_full = xr.open_dataset(out_path).load()
        ds_full.close()
        # append along diff values for step
        ds_new = xr.Dataset.from_dict(self.ds_storage)

        ds = xr.concat([ds_full, ds_new], dim="step", data_vars="minimal")

        ds.to_netcdf(out_path, compute=False)
        # futures = client.compute(values)

        # update step number on dict storage
        self.ds_storage["coords"]["step"]["data"] += self.n_chunk

    def write_probable_model(self, out_path, start_time, n_bins=100):
        """
        read in file to use all saved samples.
        only use samples past burn in to compute most probable model.

        :param out_dir:
        :param n_bins:
        """
        # *** change later to use context manager ***
        ds_full = xr.open_dataset(out_path).load()
        ds_full.close()

        # get the most probable model params
        model_params = ds_full["model_params"][:, self.n_burn :]

        prob_params = np.empty(self.model_params.n_model_params)
        for p_ind in range(self.model_params.n_model_params):
            # loop over each model param and find most proable value
            counts, bins = np.histogram(model_params[p_ind], bins=n_bins, density=True)
            ind = np.argmax(counts)
            prob_value = (bins[ind] + bins[ind + 1]) / 2
            prob_params[p_ind] = prob_value

        # run forward problem to get predicted data for most probable model
        data_pred, prob_params = self.model_params.forward_model(
            self.data.periods, prob_params
        )

        # save probable params and data pred
        ds_full["prob_params"] = ("n_model_params", prob_params)
        ds_full["data_prob"] = ("period", data_pred)

        # save the inversion computation time
        end_time = time.time()
        computation_time = end_time - start_time
        ds_full = ds_full.assign_attrs(computation_time=computation_time)

        ds_full.to_netcdf(out_path, compute=False)

    def perform_tempering_swap(self):
        """
        *** fix up later ***
        """
        # tempering exchange move
        # following: https://www.cs.ubc.ca/~nando/540b-2011/projects/8.pdf

        for ind in range(self.n_chains - 1):
            # swap temperature neighbours with probability
            # look to see if there's a better way to do more than 2 chains!

            # At a given Monte Carlo step we can update the global system by swapping the
            # configuration of the two systems, or alternatively trading the two temperatures.
            # The update is accepted according to the Metropolis–Hastings criterion with probability
            beta_1 = self.chains[ind].beta
            beta_2 = self.chains[ind + 1].beta

            if beta_1 != beta_2:
                beta_ratio = beta_2 - beta_1

                logL_1 = self.chains[ind].logL
                logL_2 = self.chains[ind + 1].logL

                logratio = beta_ratio * (logL_1 - logL_2)
                xi = np.random.rand(1)
                # *** overflow in exp ***
                if xi <= np.exp(logratio):
                    # ACCEPT SWAP
                    # swap temperatures and order in list of chains? chains should be ordered by temp

                    if beta_1 == 1 or beta_2 == 1:
                        self.swap_acc += 1

                # *** swap_tot and swapacc are for calculating swap rate.... ***
                # if beta is 1, this is the master chain?
                if beta_1 == 1 or beta_2 == 1:
                    self.swap_prop += 1
