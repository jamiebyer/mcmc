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


class Inversion:
    def __init__(
        self,
        data,
        model_params,
        n_burn,
        n_chunk,
        n_mcmc,
        beta_spacing_factor=None,
        n_chains=1,
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
        self.n_chains = n_chains
        self.initialize_chains(
            model_params,
            beta_spacing_factor,
        )

        # set initial likelihood ***

        # loop over params in dict, and add with appropriate dimensions

    def define_dataset(self, model_params, out_dir):
        # general params
        data_vars = {
            "data_obs": self.data.data_obs,
            "data_pred": (
                ["step", "n_data"],
                np.empty((self.n_chunk, self.data.n_data)),
            ),
            "logL": (["step"], np.empty(self.n_chunk)),
            "acc_rate": (
                ["step"],
                np.empty(self.n_chunk),
            ),
            "err_ratio": (
                ["step"],
                np.empty(self.n_chunk),
            ),
        }

        # step values depend on the current step, since it appends existing dataset
        # every time a chunk is saved, add to step in storage ds?
        coords = {
            "n_data": np.arange(self.data.n_data),
            "step": np.arange(self.n_chunk),
        }
        self.ds_storage = xr.Dataset(data_vars=data_vars, coords=coords)

        # loop over model params to add parameters to data_vars and coords
        # data_vars = {}
        for key, val in model_params.params_info.items():
            self.ds_storage = self.ds_storage.assign_coords(
                {"n" + key: np.arange(val["n_params"])}
            )
            self.ds_storage = self.ds_storage.assign(
                {
                    key: (
                        ["step", "n_" + key],
                        np.empty((self.n_chunk, val["n_params"])),
                    )
                }
            )

        # if the output folder doesn't exist, create it.
        path_dir = os.path.dirname(out_dir)
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)

        # if this is the first iteration, create a file to save ds_results. otherwise append the results.
        if not os.path.isfile(out_dir):
            # ds_results.to_zarr(out_dir)
            self.ds_storage.to_netcdf(out_dir)

    def get_betas(self, beta_spacing_factor):
        """
        getting beta values to use for each chain.

        :param beta_spacing_factor: determines the spacing between values of beta.

        """
        # smaller spacing will have higher acceptance rates. larger spacing will explore more of the space.
        # we want to tune the spacing so our acceptance rate is 30-50%. dTlog should be larger than 1.
        # change beta steps vals based on acceptance rate

        # *** later could tune beta_spacing_factor as the inversion runs, looking at the acceptance rate every ~10 000 steps ***

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
        beta_spacing_factor,
        # optimize_starting_model=False,
    ):
        """
        initialize each of the chains, setting starting parameters, beta values, initial rotation params

        :param beta_spacing_factor
        """

        # generate the starting models
        betas = self.get_betas(beta_spacing_factor)  # get values of beta
        chains = []
        for ind in range(self.n_chains):
            model = Model(
                deepcopy(model_params),
                betas[ind],
            )

            # initialize model params
            valid_params = False
            while not valid_params:
                # generate model params between bounds.
                test_params = np.random.uniform(
                    low=model.param_bounds[0],
                    high=model.param_bounds[1],
                    size=model.model_params.n_model_params,
                )

                velocity_model = model.model_params.get_velocity_model(test_params)

                try:
                    logL, data_pred = model.get_likelihood(velocity_model, self.data)
                    valid_params = True
                except (DispersionError, ZeroDivisionError) as _:
                    valid_params = False

            model.model_params.model_params = test_params
            model.logL = logL
            model.data_pred = data_pred

            chains.append(model)

        self.chains = chains

    # async def random_walk(
    def random_walk(
        self,
        model_params,
        proposal_distribution,
        scale_factor=[1, 1],
        save_burn_in=True,
        rotation=False,
        out_filename=None,
        sample_prior=False,
    ):
        """
        perform the main loop, for n_mcmc iterations.

        :param hist_conv: value to determine convergence.
        :param out_dir: directory where to save results.
        :param save_burn_in:
        """
        if not out_filename:
            out_dir = "./results/inversion/results-" + str(int(time.time())) + ".nc"
        else:
            out_dir = "./results/inversion/results-" + out_filename + ".nc"

        # make it a dataset from the beginning?
        # allocate space from the start
        # could use model_params to
        self.define_dataset(model_params, out_dir)

        # all chains need to be on the same step number to compare
        for n_steps in range(self.n_mcmc):
            burn_in = n_steps < self.n_burn
            # save burn in: whether or not to save samples from burn in stage
            # (labeled?)

            # checking whether to write samples
            # check if saving burn in, check if there is a chunk of n_keep samples to

            delayed_results = []
            for ind in range(self.n_chains):
                chain_model = self.chains[ind]

                # rotation matrix step needs to be before step
                if rotation:
                    chain_model.update_rotation_matrix(burn_in)

                chain_model.perturb_params(
                    self.data,
                    proposal_distribution,
                    sample_prior=sample_prior,
                )

                delayed_results.append(chain_model)
                # delayed_results.append(updated_model)

            # synchronizing the separate chains
            self.chains = delayed_results

            if self.n_chains > 1:
                # tempering exchange move
                self.perform_tempering_swap()

            # check convergence using model param hist
            # *** add back later ***

            # saving sample and write to file
            # save n_chunk samples at a time
            if save_burn_in:
                write_samples = (n_steps + 1 >= self.n_chunk) and (
                    (np.mod(n_steps + 1, self.n_chunk)) == 0
                )
            else:
                # save samples past burn-in
                write_samples = not burn_in and (np.mod(n_steps + 1, self.n_chunk)) == 0

            # store every sample; only write to file every n_chunk samples.
            self.store_samples(n_steps, self.n_chunk, out_dir, write_samples)

    def store_samples(self, n_step, n_samples, out_dir, write_samples):
        """
        write out to .zarr in chunks of size n_keep.

        :param n_step: current step number in the random walk.
        :param n_samples: number of samples being saved.
        :param write_samples: whether or not to write samples to file. (write every n_keep steps)
        :param out_dir: the output directory where to save results.
        """

        # the step size on ds_storage should be up to n_batch, but should start at n_steps for the written ds.
        n_save = n_step % n_samples

        for chain in self.chains:
            # saving the chain model with beta of 1
            if chain.beta == 1:
                # update storage dataset with new param values
                # need to loop over variables for names and inds
                # *** alternatively, save model params and split to specific variables when writing to file. ***
                for key, val in chain.model_params.params_info.items():
                    # index for param and step
                    self.ds_storage[key][{"step": n_save}] = (
                        chain.model_params.model_params[val["inds"]]
                    )

                # *** remove append later ***
                self.ds_storage["logL"][{"step": n_save}] = chain.logL
                self.ds_storage["data_pred"][{"step": n_save}] = chain.data_pred.copy()
                # self.ds_storage["beta"][{"step": n_step}] = chain.beta

                if chain.swap_acc == 0:
                    self.ds_storage["acc_rate"][{"step": n_save}] = 0
                else:
                    self.ds_storage["acc_rate"][{"step": n_save}] = chain.swap_acc / (
                        chain.swap_acc + chain.swap_rej
                    )

                if chain.swap_rej == 0:
                    self.ds_storage["err_ratio"][{"step": n_save}] = 0
                else:
                    self.ds_storage["err_ratio"][{"step": n_save}] = (
                        chain.swap_err / chain.swap_rej
                    )

        if write_samples:
            """
            append current samples to file
            """
            # *** change later to use context manager ***
            ds_full = xr.open_dataset(out_dir).load()
            ds_full.close()
            ds = xr.concat([ds_full, self.ds_storage], dim="step")
            ds.to_netcdf(out_dir)  # , append_dim="step")

            """
            with xr.open_dataset(out_dir, mode="a") as ds_full:
                # ds_results.to_zarr(out_dir, append_dim="step")
                # ds_full = xr.open_dataset(out_dir)  # , engine="netcdf4")
                
                ds = xr.concat([ds_full, ds_results], dim="step")
                # combined_ds = xr.combine_by_coords(
                #    [ds_full, ds_results]
                # )  # , coords="step")

                ds.to_netcdf(out_dir)  # , append_dim="step")
                """
            # clear stored results after saving.

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

    def check_convergence(self, n_step, hist_conv, out_dir):
        """
        check if the model has converged.

        :param n_step: number of mcmc steps that have happened.
        :hist_conv: value determining model convergence.
        :out_dir: path for where to save results.
        """
        # do at least n_rot steps after starting rotation before model can converge
        enough_rotations = n_step > (self.n_burn + self.n_rot)

        """
        Check convergence for one chain, and for any number.
        """

        # *** validate. i think this is finding convergence for 2 chains. i want to generalize this. ***
        # find the max of abs of the difference between 2 models, right now hard-coded for 2 chains
        hist_diff = (
            np.abs(
                self.chains[0].model_hist / self.chains[0].model_hist.max()
                - self.chains[1].model_hist / self.chains[1].model_hist.max()
            )
        ).max()

        if (hist_diff < hist_conv) & enough_rotations:
            # store the samples up to the convergence.
            n_samples = (n_step - self.n_burn) % self.n_keep
            self.store_samples(
                hist_diff,
                n_step,
                n_samples,
                out_dir,
                write_samples=True,
            )

            # model has converged.
            sys.exit("Converged, terminate.")

        return hist_diff
