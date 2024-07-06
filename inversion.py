import numpy as np
import pandas as pd


class Inversion:
    def __init__(
        self,
        phase_vel_obs,
        chains,
        bounds,
        n_freqs,
        n_burn=10000,  # index for burning
        n_keep=2000,  # index for writing it down
        n_rot=40000,  # Do at least n_rot steps after nonlinear rotation starts
    ):
        """ """
        self.phase_vel_obs = phase_vel_obs
        self.bounds = bounds
        self.n_freqs = n_freqs
        self.n_burn = n_burn
        self.n_keep = n_keep
        self.n_rot = n_rot
        self.n_mcmc = 100000 * n_keep  # number of random walks

        self.chains = chains
        self.n_chains = len(self.chains)
        self.logL = np.zeros(self.n_chains)

        # make vector form of bounds
        self.bounds = list(bounds.values())  # should maintain order
        # add range as a third column
        # bounds array is min, max, range for each param
        np.append(
            self.bounds, (self.bounds[1, :] - self.bounds[0, :]), axis=0
        )  # verify this!

    def run_inversion(self, freqs, bounds, sigma_pd, lin_rot=True):
        if lin_rot:
            for chain_model in self.chains:
                chain_model.lin_rot(freqs, bounds, sigma_pd)

        # mcmc random walk
        self.random_walk()

    def random_walk(
        self,
        cconv=0.3,
        n_burn_in=10000,  # index for burning
        n_keep=2000,  # index for writing it down
        n_chain_thin=10,
        n_after_rot=40000,  # Do at least N_AFTER_ROT steps after nonlinear rotation started
    ):
        # initialize saved results on model (maybe should be under velocity_model?)
        for chain_model in self.chains:
            chain_model.saved_results = {
                "logL": np.zeros(n_keep),
                "m": np.zeros(n_keep),
                "d": np.zeros(n_keep),
                "acc": np.zeros(n_keep),
            }
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
        if c_diff < cconv:
            rotation = True

        save_cov_mat = False
        for n_steps in range(self.n_mcmc):
            # rotation ind should prolly be on inversion class

            if n_steps > n_burn_in and rotation is False:
                rotation = True
            if n_steps > n_keep + n_burn_in:
                save_cov_mat = True
                # mcsum[:, :, ichain] = 0.0
                # mmsum[:, ichain] = 0.0

            # will n_steps ever need to be separate for each model
            for chain_model in self.chains:
                chain_model.perturb_params()
                # cov mat on inversion or model class?
                chain_model.update_covariance_matrix(self.n_params)
                # what is cov mat used for if not saved
                if save_cov_mat:
                    chain_model.get_hist()

                ## Saving sample into buffer
                if (np.mod(n_steps, n_chain_thin)) == 0:
                    chain_model.saved_results["logL"][ikeep] = chain_model.logL
                    chain_model.saved_results["m"][ikeep] = chain_model.params
                    # saved_results["d"][ikeep, ichain] = chain_model.logL
                    chain_model.saved_results["acc"][ikeep] = float(
                        iacc[ichain, 0]
                    ) / float(iprop[ichain, 0])

                    if imcmc >= n_burn_in:
                        ikeep += 1  # keeping track of the idex to write into a file

            # SUBTRACTING 2 NORMALIZED HISTOGRAM
            if imcmc > n_burn_in:  # after burning period
                rotate = i_after_rot > n_after_rot
                self.check_convergence(self.chains, n_burn_in, n_after_rot, rotate)

    def check_convergence(chains, n_burn_in, n_after_rot, rotate, hist_conv=0.05):
        out_dir = "./out/"

        # SUBTRACTING 2 NORMALIZED HISTOGRAM
        # find the max of abs of the difference between 2 models
        # how to determine convergence with more than 2 chains?

        # right now hard-coded for 2 chains
        hist_diff = (
            np.abs(
                chains[0].hist_m / chains[0].hist_m.max()
                - chains[1].hist_m / chains[1].hist_m.max()
            )
        ).max()

        if (hist_diff < hist_conv) & rotate:

            # print("Nchain models have converged, terminating.")
            # print("imcmc: " + str(imcmc))

            # collect results
            keys = ["logL", "m", "d", "acc"]
            # logLkeep2, mkeep2, dkeep2, acc, hist_d_plot, covariance matrix
            for key in keys():
                df_dict = {}
                for ind in range(len(chains)):
                    df_dict[key] = chains[ind].saved_results[key]
                df = pd.DataFrame(df_dict)
                df.to_csv(
                    out_dir + key + ".csv",
                )

            print("saving output files %.1d %% " % (float(imcmc) / NMCMC * 100))
            print("Rotation is %s" % ("on" if (irot) else "off"))
            tend = time.time()  # starting time to keep
            print("time to converge: %s sec" % (round(tend - tstart, 2)))

            # TERMINATE!
            sys.exit("Converged, terminate.")

        elif (np.mod(imcmc, 5 * n_keep)) == 0:
            t2 = time.time()
            print("Not converging yet; time: ", t2 - t1, "s")
            print("imcmc: " + str(imcmc))
            print("hist_diff: %1.3f, cov_diff: %1.3f" % (hist_diff, c_diff))

            hist_d_plot.append(hist_diff)
            plt.figure(100)
            plt.plot(hist_d_plot, "-k")
            plt.pause(0.00001)
            plt.draw()
