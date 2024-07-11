import numpy as np
import pandas as pd
import sys
from mpi4py import MPI
import pickle


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

        # make vector form of bounds
        self.bounds = list(bounds.values())  # should maintain order
        # add range as a third column
        # bounds array is min, max, range for each param
        np.append(
            self.bounds, (self.bounds[1, :] - self.bounds[0, :]), axis=0
        )  # verify this!

    def schedule_tempering(self, rank, dTlog=1.15):
        # Parallel tempering schedule:
        n_chains_frac = int(np.ceil(self.n_chains / 4))
        n_temps = self.n_chains - n_chains_frac + 1
        Nchaint = np.zeros(n_temps, dtype=int)
        beta_pt = np.zeros(self.n_chains, dtype=float)
        Nchaint[:] = 1
        Nchaint[0] = n_chains_frac

        for it in range(n_temps):
            for ic in range(Nchaint[it]):
                for it2 in range(n_temps):
                    beta_pt[it2] = 1.0 / dTlog ** float(it)
                    # T = 1.0 / beta_pt[it2]

        return beta_pt

    def run_inversion(self, freqs, bounds, sigma_pd):
        # setup
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        status = MPI.Status()

        beta_pt = self.schedule_tempering()

        comm.Barrier()

        if rank == 0:
            # split into functions
            # master
            beta_chain = 1.0

            for n_steps in np.arange(self.n_mcmc):
                ## Receive from workers
                source_1 = status.source
                comm.Recv(
                    received_chains,
                    source=MPI.ANY_SOURCE,
                    tag=MPI.ANY_TAG,
                    status=status,
                )
                comm.Recv(prop_acc, source=source_1, tag=MPI.ANY_TAG)

                prop[source_1 - 1, :] = prop_acc[0, :]
                acc[source_1 - 1, :] = prop_acc[1, :]

                comm.Recv(
                    received_chains,
                    source=MPI.ANY_SOURCE,
                    tag=MPI.ANY_TAG,
                    status=status,
                )
                source_2 = status.source
                comm.Recv(prop_acc, source=source_2, tag=MPI.ANY_TAG)

                prop[source_2 - 1, :] = ipropacc[0, :]
                acc[source_2 - 1, :] = ipropacc[1, :]

                ## Tempering exchange move
                self.perform_tempering_swap()

                ## Sending back to workers after swap
                for chain in received_chains:
                    msend = np.append(np.array([logLpair[0], betapair[0]]), mpair[:, 0])
                    comm.Send(msend, dest=isource0, tag=rank)

                msend = np.append(np.array([logLpair[0], betapair[0]]), mpair[:, 0])
                comm.Send(msend, dest=source_1, tag=rank)
                msend = np.append(np.array([logLpair[1], betapair[1]]), mpair[:, 1])
                comm.Send(msend, dest=source_2, tag=rank)

                self.check_convergence()

        else:
            # worker
            beta_chain = beta_pt[rank - 1]
            # lin_rot
            # is lin_rot just for an initial u, pscd?
            for chain_model in self.chains:
                chain_model.lin_rot(freqs, bounds, sigma_pd)
            # mcmc random walk
            self.random_walk(comm, status)

    def random_walk(
        self,
        comm,
        status,
        cconv=0.3,
    ):
        """
        The burn-in stage performs this search in a part of the algorithm that disregards detailed balance.
        Once burn-in is concluded, sampling with detailed balance starts and samples are recorded.
        """
        # initialize saved results on model (maybe should be under velocity_model?)
        for chain_model in self.chains:
            chain_model.saved_results = {
                "logL": np.zeros(self.n_keep),
                "m": np.zeros(self.n_keep),
                "d": np.zeros(self.n_keep),
                "acc": np.zeros(self.n_keep),
            }

        # *** is the correlation matrix even used rn? ***
        # move correlation matrix to velocity model
        correlation_mat = np.zeros(
            (self.n_fr, self.n_layers, self.n_chains), dtype=float
        )
        correlation_mat[:, :, 1] += 1
        c_diff = np.max(
            np.max(
                np.abs(self.correlation_mat[:, :, 0] - self.correlation_mat[:, :, 1])
            )
        )

        rotation = False
        save_cov_mat = False
        for n_steps in range(self.n_mcmc):
            # for parallel, receive info from workers
            ## Receive from workers
            comm.Recv(msend, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            isource0 = status.source
            comm.Recv(ipropacc, source=isource0, tag=MPI.ANY_TAG)

            received_chains = pickle.loads(msend)

            mpair[:, 0] = msend[2:]
            logLpair[0] = msend[0]
            betapair[0] = msend[1]
            # print('imcmc',imcmc)
            # print('isource0',isource0,msend)
            prop[isource0 - 1, :] = ipropacc[0, :]
            acc[isource0 - 1, :] = ipropacc[1, :]

            # after n_burn_in steps, start using PC rotation for model
            # do n_burn_in steps before saving models; this is used to determine good sampling spacing
            if n_steps == self.n_burn_in and rotation is False:
                rotation = True

            # want n_keep samples before starting rotation???
            if n_steps > self.n_keep + self.n_burn_in:
                # print('Starting PC sampling with nonlinear estimate at imcmc=',imcmc)
                save_cov_mat = True
                # mcsum[:, :, ichain] = 0.0
                # mmsum[:, ichain] = 0.0

            for chain_model in received_chains:
                # maybe perturb_params should be on inversion class?
                # evolve model forward by perturbing each parameter and accepting/rejecting new model based on MH criteria
                chain_model.perturb_params(
                    self.bounds, self.layer_bounds, self.phase_vel_obs
                )
                # cov mat on inversion or model class?
                chain_model.update_covariance_matrix(self.n_params)
                # what is cov mat used for if not saved
                if save_cov_mat:
                    chain_model.get_hist()

                ## Sending model to Master
                # can I send an object through here
                # msend = np.append(np.array([logLcur, beta_chain]), mcur)

                comm.Send(pickle.dumps(received_chains), dest=0, tag=rank)
                comm.Send(self.prop_acc, dest=0, tag=rank)

                ## Receiving back from Master
                comm.Recv(msend, source=0, tag=MPI.ANY_TAG)
                mcur = msend[2:]
                logLcur = msend[0]

                if self.imcmc >= self.n_burn_in:
                    self.ikeep += 1  # keeping track of the idex to write into a file

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

    def check_convergence(self, n_steps, hist_conv=0.05, out_dir="./out/"):
        """
        check if the model has converged.

        :param n_steps: number of mcmc steps that have happened.
        :hist_conv:
        :out_dir: path for where to save results.
        """
        # do at least n_after_rot steps after starting rotation before model can converge
        enough_rotations = n_steps > (self.n_burn_in + self.n_after_rot)

        # only save after burn-in
        if n_steps > self.n_burn_in:
            save_hist = False
        else:
            # save a subset of the models
            save_hist = (np.mod(n_steps, 5 * self.n_keep)) == 0

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

        # saving sample
        if save_hist:
            self.hist_diff_plot.append(hist_diff)
            self.save_samples(n_steps)

    def save_samples(betapair):
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
