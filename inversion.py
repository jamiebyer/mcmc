import numpy as np


class Inversion:
    def __init__(
        self,
        avg_vs_obs,
        starting_model,
        bounds,
        n_layers,
        n_chains=2,
        n_burn=10000,  # index for burning
        n_keep=2000,  # index for writing it down
        n_rot=40000,  # Do at least n_rot steps after nonlinear rotation starts
        n_bins=200,
    ):
        # TODO:
        # add in burning, keeping, rotation indices.
        """ """
        self.avg_vs_obs = avg_vs_obs
        self.bounds = bounds
        self.n_layers = n_layers
        self.n_chains = n_chains
        self.n_burn = n_burn
        self.n_keep = n_keep
        self.n_rot = n_rot
        self.n_mcmc = 100000 * n_keep  # number of random walks
        self.n_bins = n_bins

        self.chains = np.fill(n_chains, starting_model)
        self.logL = np.zeros(n_chains)

        # make vector form of bounds
        self.bounds = list(bounds.values())  # should maintain order
        # add range as a third column
        # bounds array is min, max, range for each param
        np.append(
            self.bounds, (self.bounds[1, :] - self.bounds[0, :]), axis=0
        )  # verify this!

        hist_m = np.zeros(
            (self.n_bins + 1, Npar, Nchain), dtype=float
        )  # initialize histogram of model parameter, 10 bins -> 11 edges by Npar
        mnew = np.zeros((Npar, Nchain), dtype=float)
        mcur = np.zeros((Npar, Nchain), dtype=float)
        mbar = np.zeros((Npar, Nchain), dtype=float)
        dnew = np.zeros((Ndat, Nchain), dtype=float)
        # initialize dnew using both Vp and Vs
        dcur = np.zeros((Ndat, Nchain), dtype=float)
        # initialize dcur using both Vp and Vs

        correlation_mat = np.zeros(
            (self.n_layers, self.n_layers, self.n_chains), dtype=float
        )
        correlation_mat[:, :, 1] += 1

    def run_inversion(self, lin_rot=True):
        # instead of lin_rot, use a PC package ??
        if lin_rot:  # ...
            for chain_model in self.chains:
                chain_model.lin_rot()

        # mcmc random walk
        resulting_model = self.random_walk()

        # ...

    def calculate_covariance_matrix(self, model_cur, covariance_matrix, rotation: bool):
        """
        np.cov: A 1-D or 2-D array containing multiple variables and observations.
        Each row of m represents a variable, and each column a single observation of all those variables.
        """

        # normalizing
        mw = (model_cur - self.bounds[0, :]) / self.bounds[2, :]

        model_cur.mean_sum += mw  # calculating the sum of mean(m)

        model_cur.ncov += 1

        mbar[:, ichain] = mmsum[:, ichain] / model_cur.ncov

        mcsum[:, :, ichain] = mcsum[:, :, ichain] + np.outer(
            np.transpose(mw - mbar[:, ichain]), mw - mbar[:, ichain]
        )

        Cov[:, :, ichain] = (
            mcsum[:, :, ichain] / model_cur.ncov
        )  # calculating covariance matrix
        for ipar in range(self.n_params):
            for jpar in range(Npar):
                covariance_matrix[ipar, jpar, ichain] = Cov[
                    ipar, jpar, ichain
                ] / np.sqrt(Cov[ipar, ipar, ichain] * Cov[jpar, jpar, ichain])
        # rotation
        if rotation:
            u[:, :, ichain], s, vh = np.linalg.svd(
                Cov[:, :, ichain]
            )  # rotate it to its Singular Value Decomposition
            pcsd[:, ichain] = np.sqrt(s)

    def random_walk(self, cconv=0.3):
        # move correlation matrix to velocity model
        c_diff = np.max(
            np.max(
                np.abs(self.correlation_mat[:, :, 0] - self.correlation_mat[:, :, 1])
            )
        )
        rotation = False
        if c_diff < cconv:
            rotation = True

        save_cov_mat = False
        for i in range(self.n_mcmc):
            # rotation ind should prolly be on inversion class

            if n_steps > NBURNIN and rotation is False:
                rotation = True
                n_steps = 0
            if n_steps > NKEEP and rotation is True:
                save_cov_mat = True
                n_steps = 0
                # mcsum[:, :, ichain] = 0.0
                # mmsum[:, ichain] = 0.0

            # will n_steps ever need to be separate for each model
            for chain_model in self.chains:
                chain_model.perturb_params()
                covariance_matrix = self.calculate_covariance_matrix()

                if save_cov_mat:
                    # calculate the difference between ichain parameters
                    # calculating histograms

                    for ipar in np.arange(len(mt)):
                        edge = np.linspace(minlim[ipar], maxlim[ipar], nbin + 1)
                        idx_dif = np.argmin(abs(edge - mcur[ipar, ichain]))
                        hist_m[idx_dif, ipar, ichain] += 1

                ## Saving sample into buffer
                if (np.mod(imcmc, NCHAINTHIN)) == 0:
                    logLkeep2[ikeep, :] = np.append(logLcur[ichain], ichain)
                    mkeep2[ikeep, :] = np.append(mcur[:, ichain], ichain)
                    dkeep2[ikeep, :] = np.append(dcur[:, ichain], ichain)
                    acc[ikeep, :] = np.append(
                        float(iacc[ichain, 0]) / float(iprop[ichain, 0]), ichain
                    )  # Using a stride concept: first row is when ichain = 0
                    # print((iacc[ichain,:]).astype(float)/(iprop[ichain,:]).astype(float))
                    if imcmc >= NBURNIN:
                        ikeep += 1  # keeping track of the idex to write into a file

                n_steps += 1

    def check_convergence():
        # SUBTRACTING 2 NORMALIZED HISTOGRAM
        if imcmc > NBURNIN:  # after burning period
            hist_dif = (
                np.abs(
                    hist_m[:, :, 0] / hist_m[:, :, 0].max()
                    - hist_m[:, :, 1] / hist_m[:, :, 1].max()
                )
            ).max()  # find the max of abs of the difference between 2 models
            if (hist_dif < hconv) & (i_after_rot > N_AFTER_ROT):
                print("Nchain models have converged, terminating.")
                print("imcmc: " + str(imcmc))

                df = pd.DataFrame(logLkeep2, columns=["logLkeep2", "ichain"])
                df.to_csv(
                    out_dir + "logLkeep2_example.csv",
                    mode="w" if (ihead == 1) else "a",
                    header=True if (ihead == 1) else False,
                )  # save logLkeep2 to csv
                df = pd.DataFrame(mkeep2, columns=np.append(mt_name, "ichain"))
                df.to_csv(
                    out_dir + "mkeep2_example.csv",
                    mode="w" if (ihead == 1) else "a",
                    header=True if (ihead == 1) else False,
                )  # save mkeep2 to csv
                df = pd.DataFrame(dkeep2, columns=np.append(sta_name_all, "ichain"))
                df.to_csv(
                    out_dir + "dkeep2_example.csv",
                    mode="w" if (ihead == 1) else "a",
                    header=True if (ihead == 1) else False,
                )  # save dkeep2 to csv
                df = pd.DataFrame(acc, columns=np.append("acc", "ichain"))
                df.to_csv(
                    out_dir + "acc_example.csv",
                    mode="w" if (ihead == 1) else "a",
                    header=True if (ihead == 1) else False,
                )  # save acc to csv
                df = pd.DataFrame(hist_d_plot)
                df.to_csv(
                    out_dir + "Conv_example.csv",
                    mode="w" if (ihead == 1) else "a",
                    header=True if (ihead == 1) else False,
                )

                for ichain_id in np.arange(Nchain):
                    df = pd.DataFrame(Cov[:, :, ichain_id])
                    df.to_csv(
                        out_dir + "Cov_example_" + str(ichain_id) + ".csv"
                    )  # save covariance matrix to csv

                print("saving output files %.1d %% " % (float(imcmc) / NMCMC * 100))
                print("Rotation is %s" % ("on" if (irot) else "off"))
                tend = time.time()  # starting time to keep
                print("time to converge: %s sec" % (round(tend - tstart, 2)))

                # TERMINATE!
                sys.exit("Converged, terminate.")

            elif (np.mod(imcmc, 5 * NKEEP)) == 0:
                t2 = time.time()
                print("Not converging yet; time: ", t2 - t1, "s")
                print("imcmc: " + str(imcmc))
                print("hist_diff: %1.3f, cov_diff: %1.3f" % (hist_dif, c_diff))

                hist_d_plot.append(hist_dif)
                plt.figure(100)
                plt.plot(hist_d_plot, "-k")
                plt.pause(0.00001)
                plt.draw()
