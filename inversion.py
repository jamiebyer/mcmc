import numpy as np
import pandas as pd
import sys
import dask
from model import ChainModel
from dask.distributed import Client
import dask.dataframe as dd
import os
import xarray as xr

"""
TODO:
- add back NCHAINTHIN (randomized?)
- order of params..?
- rotation
- save acceptance rate
- change beta steps vals based on acceptance rate
"""


class Inversion:
    def __init__(
        self,
        n_data,
        param_bounds,
        freqs,
        phase_vel_obs,
        poisson_ratio=0.265,
        density_params=[540.6, 360.1],  # *** check units
        n_chains=2,
        n_layers=10,
        n_bins=200,
        n_burn_in=10000,
        n_keep=2000,  # index for writing it down
        n_rot=40000,  # Do at least n_rot steps after nonlinear rotation starts
    ):
        """
        :param freqs: frequencies at which data is measured
        :param n_layers: number of layers in the model
        :param param_bounds: min, max bounds for each model param
        :param starting_models: starting model for each chain
        :param phase_vel_obs: observed data/phase velocity
        :param n_bins: number of bins for the model histograms
        :param n_burn_in: number of steps to discard from the start of the run (to avoid bias towards the starting model)
        :param n_keep: number of steps/iterations to save to file at a time. determines total number of steps.
        :param n_rot: number of steps to do after nonlinear rotation starts
        """
        # parameters from scene
        self.freqs = freqs
        self.n_layers = n_layers
        self.phase_vel_obs = phase_vel_obs  # observed data

        # parameters related to number of steps taken in random walk
        self.n_burn_in = n_burn_in
        self.n_keep = n_keep
        self.n_rot = n_rot
        self.n_mcmc = 100000 * n_keep  # number of steps for the random walk

        # define chains here
        self.n_chains = n_chains
        self.initialize_chains(
            n_data, n_bins, param_bounds, poisson_ratio, density_params
        )

        # parameters for saving data
        self.n_bins = n_bins
        self.hist_diff_plot = []

        self.swap_acc = 0
        self.swap_prop = 0

        self.stored_results = {
            "params": [],
            "logL": [],
            "beta": [],
            "rot_mat": [],
            "sigma_pd": [],
        }

    def get_betas(self, dTlog=1.15):
        """
        setting values for beta to be used. the first quarter of the chains have beta=0

        :param dTlog: determines the spacing between values of beta. smaller spacing will
        have higher acceptance rates. larger spacing will explore more of the space. we want to tune
        the spacing so our acceptance rate is 30-50%. dTlog should be larger than 1.

        :return beta: inverse temperature; beta values to use for each chain
        """
        # *** maybe this function should be replaced with a property later***
        # Parallel tempering schedule
        n_temps = self.n_chains
        # *** later could tune dTlog as the inversion runs, looking at the acceptance rate every ~10 000 steps ***
        # 1/4 to 1/2 of the chains should be beta=1.
        n_temps_frac = int(np.ceil(n_temps / 4))
        betas = np.zeros(n_temps, dtype=float)

        inds = np.arange(n_temps_frac, n_temps)
        betas[inds] = 1.0 / dTlog**inds
        # T = 1.0 / beta
        betas = [0.5, 1]  # *** hard coded beta for now ***
        return betas

    def initialize_chains(
        self, n_data, n_bins, param_bounds, poisson_ratio, density_params
    ):
        # generate the starting models
        betas = self.get_betas()
        chains = []
        for ind in range(self.n_chains):
            # *** generate params within bounds, poisson_ratio, and density_params ? ***
            # *** starting_model params should be separate from true model params ***
            model = ChainModel(
                betas[ind],
                self.phase_vel_obs,
                n_bins,
                self.n_layers,
                n_data,
                self.freqs,
                param_bounds,
                poisson_ratio,
                density_params,
            )
            # *** validate sigma_pd ***
            # setting initial values for u, pcsd...
            # *** lin rot is updated every iteration in burn in? ***
            # *** choosing variance (trial and error...?) ***
            model.lin_rot(variance=12)

            chains.append(model)
        self.chains = chains

    async def random_walk(self, hist_conv, out_dir):
        """
        :param hist_conv:
        """
        async with Client(asynchronous=True) as client:
            for n_steps in range(self.n_mcmc):
                print("\n", n_steps)
                save_burn_in = True
                update_cov_mat = n_steps >= self.n_burn_in
                if save_burn_in:
                    save_samples = True
                    write_samples = (n_steps >= self.n_keep - 1) and (
                        (np.mod(n_steps + 1, self.n_keep)) == 0
                    )
                    update_rot_mat = (
                        write_samples
                        and (n_steps > self.n_burn_in)
                        and n_steps != self.n_keep
                    )
                    end_burn_in = n_steps == self.n_keep - 1  # for test, save burn-in
                else:
                    save_samples = update_cov_mat
                    # n_keep doesn't need to be same freq as updating rot_mat
                    write_samples = (n_steps >= self.n_burn_in - 1) and (
                        np.mod(n_steps + 1, self.n_keep)
                    ) == 0
                    update_rot_mat = write_samples and n_steps != self.n_keep
                    end_burn_in = n_steps == self.n_burn_in - 1

                # PARALLEL COMPUTING
                delayed_results = []
                for ind in range(self.n_chains):
                    chain_model = self.chains[ind]

                    updated_model = client.submit(
                        self.perform_step, chain_model, update_cov_mat, update_rot_mat
                    )
                    delayed_results.append(updated_model)

                # synchronizing the separate chains
                self.chains = await client.gather(delayed_results)

                # prop[source_1 - 1, :] = prop_acc[0, :]
                # acc[source_1 - 1, :] = prop_acc[1, :]

                # Tempering exchange move
                self.perform_tempering_swap()

                # saving sample and write to file
                hist_diff = self.check_convergence(n_steps, hist_conv, out_dir)

                self.store_samples(
                    hist_diff, n_steps, self.n_keep, out_dir, write_samples, end_burn_in
                )

    async def perform_step(self, chain_model, update_cov_mat, update_rot_mat):
        """
        update one chain model.
        perturb each param on the chain model and accept each new model with a likelihood.

        :param chain_model:

        :return: updated chain model
        """
        # *** might move perturb params to inversion class. clean this up ***
        # evolve model forward by perturbing each parameter and accepting/rejecting new model based on MH criteria
        chain_model.perturb_params()

        # start saving cov_mat after burn-in
        # *** can we get the covariance matrix for both chains at the same time? ***
        if update_cov_mat:
            chain_model.update_covariance_matrix(update_rot_mat)

        # only update the histogram if it's being saved
        # *** need to add hist to saved results? ***
        chain_model.update_model_hist()

        return chain_model

    def perform_tempering_swap(self):
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
        enough_rotations = n_step > (self.n_burn_in + self.n_rot)

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
            n_samples = (n_step - self.n_burn_in) % self.n_keep
            self.store_samples(
                hist_diff,
                n_step,
                n_samples,
                out_dir,
                write_samples=True,
                create_file=False,
            )

            # model has converged.
            sys.exit("Converged, terminate.")

        return hist_diff

    def store_samples(
        self, hist_diff, n_step, n_samples, out_dir, write_samples, create_file
    ):
        """
        write out to .zarr in chunks of size n_keep.

        :param hist_diff: used for determining convergence.
        :param n_step: current step number in the random walk.
        :param n_samples: number of samples being saved.
        :param write_samples: whether or not to write samples to file. (write every n_keep steps)
        :param create_file: whether or not this is the first save to file, and if the file needs to be created.
        :param out_dir: the output directory where to save results.
        """
        for chain in self.chains:
            # *** how to save model_hist? ***
            chain.update_model_hist()
            # saving the chain model with beta of 1
            # *** also there are multiple chains like that. which one do i save? ***
            if chain.beta == 1:
                # maybe move this to model class
                self.stored_results["params"].append(chain.model_params)
                self.stored_results["logL"].append(chain.logL)
                self.stored_results["beta"].append(chain.beta)
                self.stored_results["rot_mat"].append(chain.rot_mat)
                self.stored_results["sigma_pd"].append(chain.sigma_pd)
                self.stored_results["hist_diff"].append(hist_diff)
                # self.stored_results["acc"].append(chain.acc)

        if write_samples:
            # *** i don't like the way this is getting n_params. ***
            n_params = len(self.stored_results["params"][0])
            # create dataset to store results
            ds_results = xr.Dataset(
                coords={
                    "step": np.zeros(n_samples),
                    "param": np.arange(n_params),
                },
            )
            ds_results["step"] = np.arange(n_step + 1 - n_samples, n_step + 1)

            # add model params, logL, sigma_pd, rot_mat to the results dataset
            ds_results["params"] = (
                ["step", "param"],
                self.stored_results["params"],
            )
            ds_results["logL"] = (["step"], self.stored_results["logL"])
            ds_results["sigma_pd"] = (["step"], self.stored_results["sigma_pd"])
            ds_results["rot_mat"] = (
                ["step", "param", "param"],
                self.stored_results["rot_mat"],
            )

            # if the output folder doesn't exist, create it.
            path_dir = os.path.dirname(out_dir)
            if not os.path.isdir(path_dir):
                os.mkdir(path_dir)

            # if this is the first iteration, create a file to save ds_results. otherwise append the results.
            if create_file:
                ds_results.to_zarr(out_dir)
            else:
                ds_results.to_zarr(out_dir, append_dim="step")

            # clear stored results after saving.
            self.stored_results = {
                "params": [],
                "logL": [],
                "beta": [],
                "rot_mat": [],
                "sigma_pd": [],
            }
