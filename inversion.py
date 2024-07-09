import numpy as np
import pandas as pd
import sys


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
        self.hist_diff_plot = []

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
        n_chain_thin=10,
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
            # for parallel, receive info from workers
            """
            ##
            ## Receive from workers
            ##
            comm.Recv(msend, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            isource0 = status.source
            comm.Recv(ipropacc, source=isource0, tag=MPI.ANY_TAG)
            mpair[:, 0] = msend[2:]
            logLpair[0] = msend[0]
            betapair[0] = msend[1]
            # print('imcmc',imcmc)
            # print('isource0',isource0,msend)
            prop[isource0 - 1, :] = ipropacc[0, :]
            acc[isource0 - 1, :] = ipropacc[1, :]

            comm.Recv(msend, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            isource1 = status.source
            comm.Recv(ipropacc, source=isource1, tag=MPI.ANY_TAG)
            mpair[:, 1] = msend[2:]
            logLpair[1] = msend[0]
            betapair[1] = msend[1]
            # print('isource1',isource1,msend)
            prop[isource1 - 1, :] = ipropacc[0, :]
            acc[isource1 - 1, :] = ipropacc[1, :]
            """

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

            for chain_model in self.chains:
                chain_model.perturb_params()
                # cov mat on inversion or model class?
                chain_model.update_covariance_matrix(self.n_params)
                # what is cov mat used for if not saved
                if save_cov_mat:
                    chain_model.get_hist()

                ## Saving sample into buffer
                if (np.mod(n_steps, n_chain_thin)) == 0:
                    chain_model.saved_results["logL"][self.ikeep] = chain_model.logL
                    chain_model.saved_results["m"][ikeep] = chain_model.params
                    # saved_results["d"][ikeep, ichain] = chain_model.logL
                    chain_model.saved_results["acc"][self.ikeep] = float(
                        iacc[ichain, 0]
                    ) / float(iprop[ichain, 0])

                    if self.imcmc >= self.n_burn_in:
                        self.ikeep += (
                            1  # keeping track of the idex to write into a file
                        )
            self.perform_tempering_swap()
            self.check_convergence(n_steps)
            self.save_samples(n_steps)

    def perform_tempering_swap(self):
        ##
        ## Tempering exchange move
        ##
        # print('pair0',mpair[:,0])
        # print('pair1',mpair[:,1])
        if betapair[0] != betapair[1]:
            betaratio = betapair[1] - betapair[0]
            # print('swap',betapair[1],betapair[0],betaratio,logLpair[0],logLpair[1],logLpair[0]-logLpair[1])
            logratio = betaratio * (logLpair[0] - logLpair[1])
            xi = np.random.rand(1)
            # print('logratio',xi,logratio,np.exp(logratio))
            if xi <= np.exp(logratio):
                ## ACCEPT SWAP
                #    PRINT*,'ACCEPTED',EXP(logratio*betaratio),ran_uni2
                #    WRITE(ulog,204)'ic1:',ic1idx,obj(ic1idx)%k,logP1,obj(ic1idx)%logL,logratio,betaratio,EXP(logratio*betaratio),ran_uni2
                #    WRITE(ulog,204)'ic2:',ic2idx,obj(ic2idx)%k,logP2,obj(ic2idx)%logL
                mpair[:, [0, 1]] = mpair[:, [1, 0]]
                # print('accept')
                if (betapair[0] == 1.0) | (betapair[1] == 1.0):
                    swapacc = swapacc + 1
            if (betapair[0] == 1.0) | (betapair[1] == 1.0):
                swapprop = swapprop + 1
        ## Sending back to workers after swap
        # print('pair0',mpair[:,0])
        # print('pair1',mpair[:,1])
        msend = np.append(np.array([logLpair[0], betapair[0]]), mpair[:, 0])
        comm.Send(msend, dest=isource0, tag=rank)
        msend = np.append(np.array([logLpair[1], betapair[1]]), mpair[:, 1])
        comm.Send(msend, dest=isource1, tag=rank)

    def check_convergence(self, n_steps, hist_conv=0.05, out_dir="./out/"):

        # SUBTRACTING 2 NORMALIZED HISTOGRAM
        # do at least n_after_rot steps after starting rotation before model can converge
        rotate = n_steps > (self.n_burn_in + self.n_after_rot)
        save_hist = (np.mod(self.imcmc, 5 * self.n_keep)) == 0

        # SUBTRACTING 2 NORMALIZED HISTOGRAM
        # find the max of abs of the difference between 2 models
        # right now hard-coded for 2 chains
        hist_diff = (
            np.abs(
                self.chains[0].hist_m / self.chains[0].hist_m.max()
                - self.chains[1].hist_m / self.chains[1].hist_m.max()
            )
        ).max()

        # check for convergence
        if (hist_diff < hist_conv) & rotate:
            # print("Nchain models have converged, terminating.")
            # print("imcmc: " + str(imcmc))

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

            # print("saving output files %.1d %% " % (float(imcmc) / NMCMC * 100))
            # print("Rotation is %s" % ("on" if (irot) else "off"))
            # tend = time.time()  # starting time to keep
            # print("time to converge: %s sec" % (round(tend - tstart, 2)))

            # TERMINATE!
            sys.exit("Converged, terminate.")

        if save_hist:
            # t2 = time.time()
            # print("Not converging yet; time: ", t2 - t1, "s")
            # print("imcmc: " + str(imcmc))
            # print("hist_diff: %1.3f, cov_diff: %1.3f" % (hist_diff, c_diff))

            self.hist_diff_plot.append(hist_diff)
            # plt.figure(100)
            # plt.plot(hist_d_plot, "-k")
            # plt.pause(0.00001)
            # plt.draw()

    def save_samples():
        """
        Write out to csv in chunks of size n_keep.
        """
        ##
        ## Saving sample into buffers
        ##
        if betapair[0] == 1.0:
            logLkeep2[ikeep, :] = np.append(logLpair[0], betapair[0])
            mkeep2[ikeep, :] = np.append(mpair[:, 0], isource0)
            # dkeep2[ikeep,:] = np.append(dcur[:,ichain],isource0)
            acckeep2[ikeep, :] = (acc[isource0 - 1, :]).astype(float) / (
                prop[isource0 - 1, :]
            ).astype(float)
            # print(imcmc,isource0,betapair[0])
            ikeep += 1
        if betapair[1] == 1.0:
            logLkeep2[ikeep, :] = np.append(logLpair[1], betapair[1])
            mkeep2[ikeep, :] = np.append(mpair[:, 1], isource1)
            # dkeep2[ikeep,:] = np.append(dcur[:,ichain],isource0)
            acckeep2[ikeep, :] = (acc[isource1 - 1]).astype(float) / (
                prop[isource1 - 1]
            ).astype(float)
            # print(imcmc,isource0,betapair[1])
            ikeep += 1
        #        #SUBTRACTING 2 NORMALIZED HISTOGRAM
        #        if (imcmc > Nburnin): # Save after burn-in
        #            hist_dif=((np.abs(hist_m[:,:,0]/hist_m[:,:,0].max()-hist_m[:,:,1]/hist_m[:,:,1].max())).max()) #find the max of abs of the difference between 2 models
        #            if (hist_dif < hconv) & (i_after_rot > N_AFTER_ROT):
        #                print('Nchain models have converged, terminating.')
        #                print('imcmc: '+str(imcmc))
        #
        #                df=pd.DataFrame(logLkeep2,columns=['logLkeep2','ichain'])
        #                df.to_csv(out_dir+"logLkeep2_example.csv",mode= 'w' if (ihead==1) else 'a',header= True if (ihead==1) else False) #save logLkeep2 to csv
        #                df = pd.DataFrame(mkeep2,columns=np.append(mt_name,'ichain'))
        #                df.to_csv(out_dir+"mkeep2_example.csv",mode= 'w' if (ihead==1) else 'a',header= True if (ihead==1) else False) #save mkeep2 to csv
        #                df=pd.DataFrame(dkeep2,columns=np.append(sta_name_all,'ichain'))
        #                df.to_csv(out_dir+"dkeep2_example.csv",mode= 'w' if (ihead==1) else 'a',header= True if (ihead==1) else False) #save dkeep2 to csv
        #                df=pd.DataFrame(acc,columns=np.append('acc','ichain'))
        #                df.to_csv(out_dir+"acc_example.csv",mode= 'w' if (ihead==1) else 'a',header= True if (ihead==1) else False) #save acc to csv
        #                df=pd.DataFrame(hist_d_plot)
        #                df.to_csv(out_dir+"Conv_example.csv",mode= 'w' if (ihead==1) else 'a',header= True if (ihead==1) else False)
        #
        #                for ichain_id in np.arange(Nchain):
        #                    df=pd.DataFrame(Cov[:,:,ichain_id])
        #                    df.to_csv(out_dir+"Cov_example_"+str(ichain_id)+".csv") #save covariance matrix to csv
        #
        #                print("saving output files %.1d %% "%(float(imcmc)/NMCMC*100))
        #                print('Rotation is %s'%('on' if (irot) else 'off') )
        #                tend=time.time() #starting time to keep
        #                print('time to converge: %s sec'%(round(tend-tstart,2)))
        #
        #                #TERMINATE!
        #                sys.exit('Converged, terminate.')
        #
        #            elif (np.mod(imcmc,5*NKEEP)) == 0:
        #                t2 = time.time()
        #                print('Runtime: ',t2-t1,'s;  imcmc: '+str(imcmc),'hist_diff: %1.3f, cov_diff: %1.3f'%(hist_dif,c_diff))
        #                hist_d_plot.append(hist_dif)
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
