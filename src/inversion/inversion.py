import numpy as np
import sys

from disba import PhaseDispersion
from disba._exception import DispersionError
from inversion.model import Model
from dask.distributed import Client
import os
import xarray as xr
import time


class Inversion:
    def __init__(
        self,
        data,
        n_layers,
        sigma_model,
        poisson_ratio,
        param_bounds,
        n_chains,
        beta_spacing_factor,
        n_bins,
        n_burn,
        n_keep,
        n_rot,
    ):
        """
        :param n_data: number of data observed.
        :param param_bounds: min, max bounds for each model param
        :param freqs: frequencies at which data is measured
        :param phase_vel_obs: observed data from true model
        :param poisson_ratio:
        :param density_params:
        :param n_layers: number of layers in model.
        :param n_chains: number of chains
        :param beta_spacing_factor:
        :param n_bins: number of bins for the model histograms
        :param n_burn_in: number of steps to discard from the start of the run (to avoid bias towards the starting model)
        :param n_keep: number of steps/iterations to save to file at a time. determines total number of steps.
        :param n_rot: number of steps to do after nonlinear rotation starts
        """

        """
        run options:
        - run with optimized starting model
        - run with burn-in
        - run with linearlization
        - run with parallel tempering/ chains
        """

        # setting up model
        self.data = data
        # self.param_bounds = Model.assemble_param_bounds(param_bounds, n_layers)
        self.param_bounds = param_bounds

        # parameters related to number of steps taken in random walk
        # should function with a burn in of 0
        self.n_burn = n_burn
        self.n_rot = n_rot

        self.n_keep = n_keep
        # self.n_mcmc = 100000 * n_keep  # number of steps for the random walk
        self.n_mcmc = 1000 * n_keep  # number of steps for the random walk

        # initialize chains, generate starting params.
        self.n_chains = n_chains
        self.initialize_chains(
            n_bins,
            self.param_bounds,
            n_layers,
            sigma_model,
            poisson_ratio,
            beta_spacing_factor,
        )

        # set initial likelihood ***

        # parameters for saving data
        self.n_bins = n_bins
        self.stored_results = {
            "thickness": [],
            "vel_s": [],
            "vel_p": [],
            "density": [],
            "data_pred": [],
            "logL": [],
            "beta": [],
            # "rot_mat": [],
            # "sigma_model": [],
            # "hist_diff": [],
            "acc_rate": [],
            "err_ratio": [],
        }

    def get_betas(self, beta_spacing_factor):
        """
        getting beta values to use for each chain.

        :param beta_spacing_factor: determines the spacing between values of beta. smaller spacing will
        have higher acceptance rates. larger spacing will explore more of the space. we want to tune
        the spacing so our acceptance rate is 30-50%. dTlog should be larger than 1.
        """
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
        n_bins,
        param_bounds,
        n_layers,
        sigma_model,
        poisson_ratio,
        beta_spacing_factor,
        optimize_starting_model=False,
    ):
        """
        initialize each of the chains, setting starting parameters, beta values, initial rotation params

        :param n_bins: number of bins for histogram of results
        :param param_bounds: bounds for the model params (min, max, range)
        :param poisson_ratio: value for poisson's ratio to pass to the chain model
        :param density_params: birch's parameters to pass to the chain model
        :param beta_spacing_factor
        """

        # generate the starting models
        betas = self.get_betas(beta_spacing_factor)  # get values of beta
        chains = []
        for ind in range(self.n_chains):
            model = Model(
                n_layers,
                poisson_ratio,
                sigma_model,
                betas[ind],
                n_bins,
            )
            # sample
            valid_params = False
            while not valid_params:
                thickness = np.random.uniform(
                    self.param_bounds["thickness"][0],
                    self.param_bounds["thickness"][1],
                    (n_layers - 1),
                )
                vel_s = np.random.uniform(
                    self.param_bounds["vel_s"][0],
                    self.param_bounds["vel_s"][1],
                    n_layers,
                )

                velocity_model, valid_params = model.get_velocity_model(
                    param_bounds, thickness, vel_s
                )
                # set initial likelihood
                try:
                    model.logL, model.data_pred = model.get_likelihood(
                        velocity_model, self.data
                    )
                except (DispersionError, ZeroDivisionError) as e:
                    valid_params = False

            model.thickness = thickness
            model.vel_s = vel_s

            # get initial model
            if optimize_starting_model:
                model_params = model.get_optimization_model(param_bounds, self.data)
                model.model_params = model_params

            chains.append(model)

        self.chains = chains

    # async def random_walk(
    def random_walk(
        self,
        max_perturbations,
        proposal_distribution,
        hist_conv,
        save_burn_in=True,
        rotation=False,
        out_filename=None,
        sample_prior=False,
    ):
        """
        perform the main loop, for n_mcmc iterations.

        :param max_perturbations:
        :param hist_conv: value to determine convergence.
        :param out_dir: directory where to save results.
        :param save_burn_in:
        """
        if not out_filename:
            out_dir = "./results/inversion/results-" + str(int(time.time())) + ".nc"
        else:
            out_dir = "./results/inversion/results-" + out_filename + ".nc"
        # all chains need to be on the same step number to compare
        # async with Client(asynchronous=True) as client:
        for n_steps in range(self.n_mcmc):
            burn_in = n_steps < self.n_burn
            # save burn in: whether or not to save samples from burn in stage
            # (diff file?, labeled?)
            # how often is the rotation matrix updated? n_rot?

            # checking whether to write samples
            # check if saving burn in, check if there is a chunk of n_keep samples to
            if save_burn_in:
                write_samples = (n_steps >= self.n_keep - 1) and (
                    (np.mod(n_steps + 1, self.n_keep)) == 0
                )
            else:
                write_samples = (n_steps >= self.n_burn - 1) and (
                    np.mod(n_steps + 1, self.n_keep)
                ) == 0

            # parallel computing
            delayed_results = []
            for ind in range(self.n_chains):
                chain_model = self.chains[ind]

                # rotation matrix step needs to be before step
                if rotation:
                    chain_model.update_rotation_matrix(burn_in)
                """
                updated_model = client.submit(
                    self.perform_step,
                    chain_model,
                    max_perturbations,
                )
                """
                updated_model = self.perform_step(
                    chain_model,
                    max_perturbations,
                    proposal_distribution,
                    sample_prior=sample_prior,
                )
                # update model param hist
                # add back in later
                # chain_model.update_model_hist()

                delayed_results.append(updated_model)
                # delayed_results.append(updated_model)

            # synchronizing the separate chains
            # self.chains = await client.gather(delayed_results)
            self.chains = delayed_results

            if self.n_chains > 1:
                # tempering exchange move
                self.perform_tempering_swap()

            # if it converges, need to save final samples

            # check convergence using model param hist
            # add back later ***
            # hist_diff = self.check_convergence(n_steps, hist_conv, out_dir)
            hist_diff = 0.5
            n_samples = self.n_keep
            # saving sample and write to file
            self.store_samples(hist_diff, n_steps, n_samples, out_dir, write_samples)
            # hist_diff, n_step, n_samples, out_dir, write_samples

    # async def perform_step(
    def perform_step(
        self,
        chain_model,
        max_perturbations,
        proposal_distribution,
        sample_prior=False,
    ):
        """
        update one chain model.
        perturb each param on the chain model and accept each new model with a likelihood.
        """
        # *** what kind of random distribution should this be, and should there be a lower bound...? ***
        n_perturbations = int(np.random.uniform(max_perturbations))
        for _ in range(n_perturbations):
            # evolve model forward by perturbing each parameter and accepting/rejecting new model based on MH criteria
            chain_model.perturb_params(
                self.param_bounds,
                self.data,
                proposal_distribution,
                sample_prior=sample_prior,
            )

        return chain_model

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

    def store_samples(self, hist_diff, n_step, n_samples, out_dir, write_samples=True):
        """
        write out to .zarr in chunks of size n_keep.

        :param hist_diff: used for determining convergence.
        :param n_step: current step number in the random walk.
        :param n_samples: number of samples being saved.
        :param write_samples: whether or not to write samples to file. (write every n_keep steps)
        :param out_dir: the output directory where to save results.
        """
        for chain in self.chains:
            # add back later ***
            # chain.update_model_hist()

            # saving the chain model with beta of 1
            # *** also there are multiple chains like that. which one do i save? ***
            if chain.beta == 1:
                # maybe move this to model class
                # self.stored_results["params"].append(chain.model_params.copy())
                self.stored_results["thickness"].append(chain.thickness.copy())
                if self.stored_results["thickness"][-1] > 0.1:
                    print("stored failed")
                self.stored_results["vel_s"].append(chain.vel_s.copy())
                self.stored_results["vel_p"].append(chain.vel_p.copy())
                self.stored_results["density"].append(chain.density.copy())
                self.stored_results["logL"].append(chain.logL)
                # self.stored_results["data_pred"].append(chain.data_pred.copy())
                self.stored_results["data_pred"].append(chain.data_pred)
                self.stored_results["beta"].append(chain.beta)
                # self.stored_results["rot_mat"].append(chain.rot_mat)
                # self.stored_results["sigma_model"].append(chain.sigma_model)
                # self.stored_results["hist_diff"].append(hist_diff)
                if chain.swap_acc == 0:
                    self.stored_results["acc_rate"].append(0)
                else:
                    self.stored_results["acc_rate"].append(
                        chain.swap_acc / (chain.swap_acc + chain.swap_prop)
                    )
                if chain.swap_prop == 0:
                    self.stored_results["err_ratio"].append(0)
                else:
                    self.stored_results["err_ratio"].append(
                        chain.swap_err / chain.swap_prop
                    )

        if write_samples:
            # *** i don't like the way this is getting n_params. ***
            # n_params = len(self.stored_results["params"][0])
            # create dataset to store results
            ds_results = xr.Dataset(
                data_vars={
                    "data_obs": self.data.data_obs,
                    "data_pred": (["step", "n_data"], self.stored_results["data_pred"]),
                    "logL": (["step"], self.stored_results["logL"]),
                    # "params": (
                    #    ["step", "param"],
                    #    self.stored_results["params"],
                    # ),
                    "thickness": (
                        ["step", "n_layer"],
                        np.concatenate(
                            (
                                self.stored_results["thickness"],
                                np.zeros((len(self.stored_results["thickness"]), 1)),
                            ),
                            axis=1,
                        ),
                    ),
                    "vel_s": (
                        ["step", "n_layer"],
                        self.stored_results["vel_s"],
                    ),
                    "vel_p": (
                        ["step", "n_layer"],
                        self.stored_results["vel_p"],
                    ),
                    "density": (
                        ["step", "n_layer"],
                        self.stored_results["density"],
                    ),
                    # "rot_mat": (
                    #    ["step", "param", "param"],
                    #    self.stored_results["rot_mat"],
                    # ),
                    # "hist_diff": (
                    #    ["step", "param"],
                    #    self.stored_results["hist_diff"],
                    # ),
                    "acc_rate": (
                        ["step"],
                        self.stored_results["acc_rate"],
                    ),
                    "err_ratio": (
                        ["step"],
                        self.stored_results["err_ratio"],
                    ),
                },
                coords={
                    # "step": np.arange(n_step + 1 - n_samples, n_step + 1),
                    "step": np.arange(n_step + 1, n_step + 1 + n_samples),
                    # "param": np.arange(n_params),
                    "n_data": np.arange(self.data.n_data),
                    "n_layer": np.arange(chain.n_layers),
                },
            )

            # if the output folder doesn't exist, create it.
            path_dir = os.path.dirname(out_dir)
            if not os.path.isdir(path_dir):
                os.mkdir(path_dir)

            # also save input model: bounds, starting model
            # if this is the first iteration, create a file to save ds_results. otherwise append the results.
            if not os.path.isfile(out_dir):
                # ds_results.to_zarr(out_dir)
                ds_results.to_netcdf(out_dir)
            else:
                # not happy with this ***
                ds_full = xr.open_dataset(out_dir).load()
                ds_full.close()
                ds = xr.concat([ds_full, ds_results], dim="step")
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
            self.stored_results = {
                # "params": [],
                "thickness": [],
                "vel_s": [],
                "vel_p": [],
                "density": [],
                "data_pred": [],
                "logL": [],
                "beta": [],
                # "rot_mat": [],
                # "sigma_model": [],
                # "hist_diff": [],
                "acc_rate": [],
                "err_ratio": [],
            }
