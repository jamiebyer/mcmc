import numpy as np
import pandas as pd
import sys
import dask


class Inversion:
    def __init__(
        self,
        chains,
        bounds,
        n_burn=10000,  # index for burning
        n_keep=2000,  # index for writing it down
        n_rot=40000,  # Do at least n_rot steps after nonlinear rotation starts
    ):
        """ """
        self.bounds = bounds
        self.n_burn = n_burn
        self.n_keep = n_keep
        self.n_rot = n_rot
        self.n_mcmc = 100000 * n_keep  # number of random walks

        self.chains = chains
        self.n_chains = len(self.chains)
        self.logL = np.zeros(self.n_chains)
        self.hist_diff_plot = []

        self.swap_acc = 1
        self.swap_prop = 1

    def run_inversion(self, freqs, bounds, sigma_pd):
        """ """
        # is lin_rot just for an initial u, pscd?
        for chain_model in self.chains:
            chain_model.lin_rot(freqs, bounds, sigma_pd)

        # mcmc random walk
        self.random_walk()

    def get_beta(self, dTlog=1.15):
        """
        setting values for beta to be used. the first quarter of the chains have beta=0

        :param dTlog:

        :return beta: inverse temperature; beta values to use for each chain
        """
        # *** maybe this function should be replaced with a property later***
        # Parallel tempering schedule
        n_temps = self.n_chains
        # the first ~1/4 of chains have beta value of 0..?

        n_temps_frac = int(np.ceil(self.n_chains / 4))
        beta = np.zeros(n_temps, dtype=float)

        inds = np.arange(n_temps_frac, n_temps)
        beta[inds] = 1.0 / dTlog**inds
        # T = 1.0 / beta

        return beta

    def random_walk(
        self,
        cconv=0.3,
    ):
        """
        The burn-in stage performs this search in a part of the algorithm that disregards detailed balance.
        Once burn-in is concluded, sampling with detailed balance starts and samples are recorded.
        """
        beta_values = self.get_beta()

        rotation = False
        for n_steps in range(self.n_mcmc):
            # after n_burn_in steps, start using PC rotation for model
            # do n_burn_in steps before saving models; this is used to determine good sampling spacing
            if n_steps == self.n_burn_in and rotation is False:
                rotation = True

            # PARALLEL COMPUTING
            delayed_results = []
            for ind in range(self.n_chains):
                chain_model = self.chains[ind]
                beta = beta_values[ind]
                updated_model = dask.delayed(self.perform_step)(chain_model, beta)
                delayed_results.append(updated_model)

                if self.n_mcmc >= self.n_burn_in:
                    self.n_keep += 1  # keeping track of the idex to write into a file

            self.chains = dask.compute(*delayed_results)
            # prop[source_1 - 1, :] = prop_acc[0, :]
            # acc[source_1 - 1, :] = prop_acc[1, :]

            ## Tempering exchange move
            self.perform_tempering_swap()

            if n_steps < self.n_burn_in:
                save_samples = False
            else:
                # save a subset of the models
                save_samples = (np.mod(n_steps, 5 * self.n_keep)) == 0

            self.check_convergence(n_steps, save_samples)

            # saving sample
            if save_samples:
                self.write_samples(n_steps)

    def perform_step(self, chain_model, beta):
        """
        update one chain model.
        perturb each param on the chain model and accept each new model with a likelihood.

        :param chain_model:
        :param beta:

        :return: updated chain model
        """
        # *** might move perturb params to inversion class. clean this up ***
        # evolve model forward by perturbing each parameter and accepting/rejecting new model based on MH criteria
        chain_model.perturb_params(self.bounds, self.phase_vel_obs)
        # cov mat on inversion or model class?
        chain_model.update_covariance_matrix(self.n_params)
        # only update the histogram if it's being saved
        chain_model.update_hist()

        return chain_model

    def perform_tempering_swap(self):
        ## Tempering exchange move
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
                if xi <= np.exp(logratio):
                    ## ACCEPT SWAP
                    # swap temperatures and order in list of chains? chains should be ordered by temp

                    if beta_1 == 1 or beta_2 == 1:
                        self.swap_acc += 1

                # swapprop and swapacc are for calculating swap rate....
                if beta_1 == 1 or beta_2 == 1:
                    self.swap_prop += 1

    def check_convergence(
        self, n_steps, save_samples, hist_conv=0.05, out_dir="./out/"
    ):
        """
        check if the model has converged.

        :param n_steps: number of mcmc steps that have happened.
        :hist_conv:
        :out_dir: path for where to save results.
        """
        # do at least n_after_rot steps after starting rotation before model can converge
        enough_rotations = n_steps > (self.n_burn_in + self.n_after_rot)

        # SUBTRACTING 2 NORMALIZED HISTOGRAM
        # find the max of abs of the difference between 2 models
        # right now hard-coded for 2 chains
        hist_diff = (
            np.abs(
                self.chains[0].hist_m / self.chains[0].hist_m.max()
                - self.chains[1].hist_m / self.chains[1].hist_m.max()
            )
        ).max()

        if (hist_diff < hist_conv) & enough_rotations:
            # collect results
            keys = ["logL", "m", "d", "acc"]
            # logLkeep2, mkeep2, dkeep2, acc, hist_d_plot, covariance matrix
            for key in keys():
                df_dict = {}
                for ind in range(self.n_chains):
                    df_dict[key] = self.chains[ind].saved_results[key]
                df = pd.DataFrame(df_dict)
                df.to_csv(
                    out_dir + key + ".csv",
                )

            # TERMINATE!
            sys.exit("Converged, terminate.")

        if save_samples:
            self.hist_diff_plot.append(hist_diff)

    def write_samples(betapair):
        """
        Write out to csv in chunks of size n_keep.
        """
        ## Saving sample into buffers
        if betapair[0] == 1:
            logLkeep2[ikeep, :] = np.append(logLpair[0], betapair[0])
            mkeep2[ikeep, :] = np.append(mpair[:, 0], isource0)
            # dkeep2[ikeep,:] = np.append(dcur[:,ichain],isource0)
            acckeep2[ikeep, :] = (acc[isource0 - 1, :]).astype(float) / (
                prop[isource0 - 1, :]
            ).astype(float)
            # print(imcmc,isource0,betapair[0])
            ikeep += 1
        if betapair[1] == 1:
            logLkeep2[ikeep, :] = np.append(logLpair[1], betapair[1])
            mkeep2[ikeep, :] = np.append(mpair[:, 1], isource1)
            # dkeep2[ikeep,:] = np.append(dcur[:,ichain],isource0)
            acckeep2[ikeep, :] = (acc[isource1 - 1]).astype(float) / (
                prop[isource1 - 1]
            ).astype(float)
            # print(imcmc,isource0,betapair[1])
            ikeep += 1

        #
        if ikeep >= NKEEP - 1:  # dump to a file
            df = pd.DataFrame(logLkeep2[: ikeep - 1, :], columns=["logLkeep2", "beta"])
            df.to_csv(
                out_dir + "logLkeep2_example.csv",
                mode="w" if (ihead == 1) else "a",
                header=True if (ihead == 1) else False,
            )  # save logLkeep2 to csv
            df = pd.DataFrame(
                mkeep2[: ikeep - 1, :], columns=np.append(mt_name, "ichain")
            )
            df.to_csv(
                out_dir + "mkeep2_example.csv",
                mode="w" if (ihead == 1) else "a",
                header=True if (ihead == 1) else False,
            )  # save mkeep2 to csv
            # df=pd.DataFrame(dkeep2,columns=np.append(sta_name_all,'ichain'))
            # df.to_csv(out_dir+"dkeep2_example.csv",mode= 'w' if (ihead==1) else 'a',header= True if (ihead==1) else False) #save dkeep2 to csv
            df = pd.DataFrame(acckeep2[: ikeep - 1, :], columns=[mt_name])
            df.to_csv(
                out_dir + "acc_example.csv",
                mode="w" if (ihead == 1) else "a",
                header=True if (ihead == 1) else False,
            )  # save acc to csv
            # df=pd.DataFrame(hist_d_plot)
            # df.to_csv(out_dir+"Conv_example.csv",mode= 'w' if (ihead==1) else 'a',header= True if (ihead==1) else False)

            # for ichain_id in np.arange(Nchain):
            #    df=pd.DataFrame(Cov[:,:,ichain_id])
            #    df.to_csv(out_dir+"Cov_example_"+str(ichain_id)+".csv") #save covariance matrix to csv
            print("imcmc:", imcmc)
            print("PT swap acceptance:", float(swapacc) / float(swapprop))
            print("In-chain acceptance", acckeep2[ikeep - 1, :])
            ihead = 0
            ikeep = 0
