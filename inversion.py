import numpy as np


class Inversion:
    def __init__(
        self,
        station_positions,
        events,
        prior_model,
        bounds,
        n_chains=2,
        n_burn=10000,  # index for burning
        n_keep=2000,  # index for writing it down
        n_rot=40000,  # Do at least n_rot steps after nonlinear rotation started
    ):
        """ """
        self.station_positions = station_positions
        self.events = events
        self.prior_model = prior_model
        self.n_chains = n_chains
        self.n_burn = n_burn
        self.n_keep = n_keep
        self.n_rot = n_rot
        self.n_mcmc = 100000 * n_keep  # number of random walks

        self.chains = np.fill(n_chains, prior_model)
        self.logL = np.zeros(n_chains)

        # make vector form of bounds
        self.bounds = list(bounds.values())  # should maintain order

    def run_inversion(self):
        """
        Solving for:
        - thickness of each layer
        - birch parameters for density profile
        - vs and vp of each layer
        - sigma s and sigma p?
        """

        # setup starting models
        self.initialize_chains(pcsd)
        # linrot
        # mcmc random walk

    def initialize_chains(self, pcsd):
        for chain_model in self.chains:
            # generate a model that fits the priors
            valid_model = False
            while not valid_model:
                # adding random normal noise to the true model for the starting model
                chain_model.generate_starting_model(self.prior_model, self.bounds, pcsd)
                if chain_model is not None:
                    valid_model = True
            chain_model.update_likelihood(self.stations, self.events)

    def calculate_covariance_matrix(covariance_matrix, rotation: bool):
        # if (c_new[ichain] == 2) & (icount[ichain] == 0):

        # normalizing
        mw = (mcur[:, ichain] - minlim) / maxpert
        ncov[ichain] += 1
        mmsum[:, ichain] = mmsum[:, ichain] + mw  # calculating the sum of mean(m)
        mbar[:, ichain] = mmsum[:, ichain] / ncov[ichain]
        mcsum[:, :, ichain] = mcsum[:, :, ichain] + np.outer(
            np.transpose(mw - mbar[:, ichain]), mw - mbar[:, ichain]
        )

        Cov[:, :, ichain] = (
            mcsum[:, :, ichain] / ncov[ichain]
        )  # calculating covariance matrix
        for ipar in range(Npar):
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

    def random_walk():
        # initialize correlation matrix, 3 dimmensions with Nchain
        correlation_mat = np.zeros((Npar, Npar, Nchain), dtype=float)
        correlation_mat[:, :, 1] += 1

        for i in range(self.n_mcmc):

            c_diff = np.max(
                np.max(np.abs(correlation_mat[:, :, 0] - correlation_mat[:, :, 1]))
            )

            if c_diff < cconv:
                rotation = True

            for chain_model in self.chains:
                # normalizing
                mtry = (mcur[:, ichain] - minlim) / maxpert
                mtry = np.matmul(np.transpose(u[:, :, ichain]), mtry)  # with rotation

                # perturb each parameter in the model
                chain_model.perturb_params()
                """
                if (icount[ichain] > NKEEP) & (
                    c_new[ichain] >= 1
                ):  # c_new 2 is for activating rotation
                    c_new[ichain] = 2
                    icount[ichain] = 0
                if (icount[ichain] > NBURNIN) & (
                    c_new[ichain] == 0
                ):  # eliminate the first # of computing covariance
                    icount[ichain] = 0
                    c_new[ichain] = 1
                    ncov[ichain] = 0.0
                    mcsum[:, :, ichain] = 0.0
                    mmsum[:, ichain] = 0.0
                
                """

                covariance_matrix = self.calculate_covariance_matrix()

                icount[ichain] += 1  # counter for rotation matrix

                if (
                    c_new[ichain] >= 1
                ):  # calculate the difference between ichain parameters
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
