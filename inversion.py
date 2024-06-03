import numpy as np



class Inversion:
    def __init__(self, station_positions, events, prior_model, n_chains=2):
        """ """
        self.station_positions = station_positions
        self.events = events
        self.prior_model = prior_model
        self.current_model = prior_model
        self.n_chains = n_chains

    def run_inversion():
        #for event in self.events:
        #    event.set_t_obs() 
        # setup starting models
        # linrot
        # mcmc random walk


    def generate_starting_model():
        while ibreak == 0:
                mnew[:, ichain] = mt + pcsd[:, ichain] * np.random.randn(len(mt))
                # Check that mtry is inside unifrom prior:
                if ((maxlim - mnew[:, ichain]).min() > 0.0) & (
                    (mnew[:, ichain] - minlim).min() > 0.0
                ):
                    # print('ichain',ichain,'mstart',mnew[:,ichain])
                    ibreak = 1
                # else:
                #    print('outside ichain')
            mcur[:, ichain] = mnew[:, ichain].copy()

    def calculate_likelihood(events, model):
        # P and S velocities for each event are all data

        for event in events:
            # calculate log(L) for each event
            t_obs_p = event.t_obs_p
            t_obs_s = event.t_obs_s

            t_model_p = model.t_p
            t_model_s = model.t_s

            vel_model_p = 0
            vel_model_s = 0

            res_p = t_obs_p - t_model_p # calculate p wave residual
            res_s = t_obs_s - t_model_s # calculate p wave residual

            n_data_p = len(res_p)
            n_data_s = len(res_s)

            
            logL_p = - (1/2) * n_data_p * np.log(mnew[-2, ichain])
            - np.sum(res_p ** 2) / (2 * mnew[-2, ichain] ** 2)

            np.sum(res_p**2)
        


        likelihood = (
            -Ndatevp / 2.0 * np.log(mnew[-2, ichain])
            - np.sum((dobs[ist : ist + Nsta] - dtmp[0:Nsta]) ** 2)
            / (2.0 * mnew[-2, ichain] ** 2)
            - Ndatevs / 2.0 * np.log(mnew[-1, ichain])
            - np.sum((dobs[ist + Nsta : iend] - dtmp[Nsta:]) ** 2)
            / (2.0 * mnew[-1, ichain] ** 2)
        )


        logLtmp[iev] = (
            -Ndatevp / 2.0 * np.log(mnew[-2, ichain])
            - np.sum((dobs[ist : ist + Nsta] - dtmp[0:Nsta]) ** 2)
            / (2.0 * mnew[-2, ichain] ** 2)
            - Ndatevs / 2.0 * np.log(mnew[-1, ichain])
            - np.sum((dobs[ist + Nsta : iend] - dtmp[Nsta:]) ** 2)
            / (2.0 * mnew[-1, ichain] ** 2)
        )

    def initialize_chains(n_chains):

        for chain in range(n_chains):

            # generate a model that fits the priors
            new_model = self.generate_starting_model()

            for event in self.n_events:
                # set dobs for each event for the new model

                # calculate likeliness function

                # get the best likelihood from the events..?
                logLkeep2[0, ichain] = np.sum(logLtmp)
                pass



            for iev in range(Nevent):
                mtmp = np.append(
                    mnew[iev * 4 : (iev + 1) * 4, ichain], mnew[-4:, ichain]
                )
                dtmp = lf.get_times(m=mtmp, xr=xr, yr=yr, zr=zr, Nsta=Nsta)
                ist = iev * Ndatev
                iend = (iev + 1) * Ndatev
                dnew[ist:iend, ichain] = dtmp
                logLtmp[iev] = (
                    -Ndatevp / 2.0 * np.log(mnew[-2, ichain])
                    - np.sum((dobs[ist : ist + Nsta] - dtmp[0:Nsta]) ** 2)
                    / (2.0 * mnew[-2, ichain] ** 2)
                    - Ndatevs / 2.0 * np.log(mnew[-1, ichain])
                    - np.sum((dobs[ist + Nsta : iend] - dtmp[Nsta:]) ** 2)
                    / (2.0 * mnew[-1, ichain] ** 2)
                )

            logLkeep2[0, ichain] = np.sum(logLtmp)

        logLcur = logLkeep2[0, :].copy()
        dcur = dnew.copy()


    def generate_model():
        pass



    def random_walk():
        for i in range(self.n_mcmc):

            c_diff = np.max(np.max(np.abs(R[:, :, 0] - R[:, :, 1])))
            if (c_diff < cconv) & irot:
                irot_start = True


            for chain in range(self.nmcmc):
                mtry = (mcur[:, ichain] - minlim) / maxpert
                mtry = np.matmul(np.transpose(u[:, :, ichain]), mtry)  # with rotation


                # perturb each parameter in the model
                for ipar in np.arange(Npar):
                    

                    ## Cauchy proposal:
                    mtry[ipar] = mtry[ipar] + 1.3 * pcsd[ipar, ichain] * np.tan(
                        np.pi * (np.random.rand(1)[0] - 0.5)
                    )
                    ## Gaussian proposal
                    #            mtry[ipar] = mtry[ipar]+ 1.1*pcsd[ipar,ichain]*np.random.randn(1)
                    mtry = np.matmul(u[:, :, ichain], mtry)
                    mtry = minlim + (mtry * maxpert)

                    iprop[ichain, ipar] += 1
                    if ((maxlim - mtry).min() > 0.0) & (
                        (mtry - minlim).min() > 0.0
                    ):  # Check that mtry is inside unifrom prior:

                        # calculate dtry from both Vp and Vs
                        # dtry = lf.get_times(m=mtry,xr=xr,yr=yr,zr=zr,Nsta=Nsta)
                        for iev in range(Nevent):
                            mtmp = np.append(mtry[iev * 4 : (iev + 1) * 4], mtry[-4:])
                            dtmp = lf.get_times(m=mtmp, xr=xr, yr=yr, zr=zr, Nsta=Nsta)
                            ist = iev * Ndatev
                            iend = (iev + 1) * Ndatev
                            dtry[ist:iend] = dtmp
                            logLtmp[iev] = (
                                -Ndatevp / 2.0 * np.log(mtry[-2])
                                - np.sum((dobs[ist : ist + Nsta] - dtmp[0:Nsta]) ** 2)
                                / (2.0 * mtry[-2] ** 2)
                                - Ndatevs / 2.0 * np.log(mtry[-1])
                                - np.sum((dobs[ist + Nsta : iend] - dtmp[Nsta:]) ** 2)
                                / (2.0 * mtry[-1] ** 2)
                            )
                        logLtry = np.sum(logLtmp)
                        #                logLtry = -Ndat/2.*np.log(mtry[-1])-np.sum((dobs-dtry)**2)/(2*mtry[-1]**2);

                        # Compute likelihood ratio in log space:
                        dlogL = logLtry - logLcur[ichain]
                        xi = np.random.rand(1)
                        # Apply MH criterion (accept/reject)
                        if xi <= np.exp(dlogL):
                            iacc[ichain, ipar] += 1
                            logLcur[ichain] = logLtry
                            mcur[:, ichain] = mtry
                            dcur[:, ichain] = dtry

                # t2=time.time()
                # print('time:',t2-t1)
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

                # print('Acc rate')
                # print(iacc[ichain,:]/iprop[ichain,:])
                # print(pcsd)
                mw = (mcur[:, ichain] - minlim) / maxpert  # for covariance
                ncov[ichain] += 1.0
                mmsum[:, ichain] = mmsum[:, ichain] + mw  # calculating the sum of mean(m)
                mbar[:, ichain] = (
                    mmsum[:, ichain] / ncov[ichain]
                )  # calculating covariance matrix
                mcsum[:, :, ichain] = mcsum[:, :, ichain] + np.outer(
                    np.transpose(mw - mbar[:, ichain]), mw - mbar[:, ichain]
                )
                if (c_new[ichain] == 2) & (icount[ichain] == 0):
                    Cov[:, :, ichain] = (
                        mcsum[:, :, ichain] / ncov[ichain]
                    )  # calculating covariance matrix
                    for ipar in range(Npar):
                        for jpar in range(Npar):
                            R[ipar, jpar, ichain] = Cov[ipar, jpar, ichain] / np.sqrt(
                                Cov[ipar, ipar, ichain] * Cov[jpar, jpar, ichain]
                            )
                    if irot_start & irot:  # ROTATION
                        u[:, :, ichain], s, vh = np.linalg.svd(
                            Cov[:, :, ichain]
                        )  # rotate it to its Singular Value Decomposition
                        pcsd[:, ichain] = np.sqrt(s)
                icount[ichain] += 1  # counter for rotation matrix

                if c_new[ichain] >= 1:  # calculate the difference between ichain parameters
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




    Ndat = 2 * Nsta * Nevent  # Total No. data
    Ndatev = 2 * Nsta  # No. data per event
    Ndatevp = Nsta  # No. data per event
    Ndatevs = Nsta  # No. data per event

    mkeep2 = np.zeros((NKEEP, Npar + 1), dtype=float)
    # initialize parameter keep (a matrix), +1 for assigning ichain
    logLkeep2 = np.zeros((NKEEP, Nchain), dtype=float)
    # initialize L2 norm keep  (a matrix?)
    logLcur = np.zeros((NKEEP, Nchain), dtype=float)
    # initialize logLcur  (a matrix?)
    logLtmp = np.zeros(Nevent)
    dkeep2 = np.zeros((NKEEP, Ndat + 1), dtype=float)
    # initialize forward model keep (a matrix), +1 for assigning ichain
    acc = np.zeros((NKEEP, 2), dtype=float)
    # initialize forward model keep (a matrix), +2 for assigning ichain
    dtry = np.zeros((Ndat), dtype=float)
    # initialize
    dtru = np.zeros((Ndat), dtype=float)
    # initialize true forward model (a vector)
    dobs = np.zeros((Ndat), dtype=float)
    # initialize d obs
    u = np.zeros((Npar, Npar, Nchain), dtype=float)  # initialize u for SVD
    s = np.zeros((Npar), dtype=float)  # initialize u for SVD
    vh = np.zeros((Npar, Npar), dtype=float)  # initialize u for SVD
    pcsd = np.zeros((Npar, Nchain), dtype=float)  # initialize pcsd for SVD
    mcsum = np.zeros(
        (Npar, Npar, Nchain), dtype=float
    )  # initialize covariance matrix sum
    Cov = np.zeros(
        (Npar, Npar, Nchain), dtype=float
    )  # initialize covariance matrix, 3 dimmensions with Nchain
    mmsum = np.zeros(
        (Npar, Nchain), dtype=float
    )  # initiazlize parameter mean vector: becareful with the dimension of Nx1 vs just N (a vector)
    hist_m = np.zeros(
        (nbin + 1, Npar, Nchain), dtype=float
    )  # initialize histogram of model parameter, 10 bins -> 11 edges by Npar
    mnew = np.zeros((Npar, Nchain), dtype=float)
    mcur = np.zeros((Npar, Nchain), dtype=float)
    mbar = np.zeros((Npar, Nchain), dtype=float)
    dnew = np.zeros((Ndat, Nchain), dtype=float)
    # initialize dnew using both Vp and Vs
    dcur = np.zeros((Ndat, Nchain), dtype=float)
    # initialize dcur using both Vp and Vs
    R = np.zeros(
        (Npar, Npar, Nchain), dtype=float
    )  # initialize correlation matrix, 3 dimmensions with Nchain
    R[:, :, 1] = R[:, :, 1] + 1.0

    pcsd[:, 0] = 1.0 / 20.0
    pcsd[:, 1] = 1.0 / 20.0
    u[:, :, 0] = np.eye(Npar)
    u[:, :, 1] = np.eye(Npar)

    ttmp = time.time()
    print("time3:", ttmp - tstart)

    ##calculate dtru
    for iev in range(Nevent):
        mtmp = np.append(mt[iev * 4 : (iev + 1) * 4], mt[-4:])
        dtmp = lf.get_times(m=mtmp, xr=xr, yr=yr, zr=zr, Nsta=Nsta)
        ist = iev * Ndatev
        iend = (iev + 1) * Ndatev
        dtru[ist:iend] = dtmp
        dobs[ist : ist + Nsta] = dtru[ist : ist + Nsta] + mt[-2] * np.random.randn(
            Nsta
        )  # adding noise corresponding to sigma_p/s to the observed values
        dobs[ist + Nsta : iend] = dtru[ist + Nsta : iend] + mt[-1] * np.random.randn(
            Nsta
        )

    # logLkeep2[0] = sum((dobs-dtru)**2)/(2*sigma**2);
    ttmp = time.time()

    #######
    ## Define starting model for the 2 MCMC chains:
    for ichain in np.arange(Nchain):
        # mnew[:,ichain] = minlim + maxpert * np.random.rand(np.size(mt)) #randomly generating initial parameter
        ibreak = 0
        while ibreak == 0:
            mnew[:, ichain] = mt + pcsd[:, ichain] * np.random.randn(len(mt))
            # Check that mtry is inside unifrom prior:
            if ((maxlim - mnew[:, ichain]).min() > 0.0) & (
                (mnew[:, ichain] - minlim).min() > 0.0
            ):
                # print('ichain',ichain,'mstart',mnew[:,ichain])
                ibreak = 1
            # else:
            #    print('outside ichain')
        mcur[:, ichain] = mnew[:, ichain].copy()
        # mcur[:,ichain] = mt.copy()
        #    dnew[:,ichain] = lf.get_times(m=mnew[:,ichain],xr=xr,yr=yr,zr=zr,Nsta=Nsta)
        for iev in range(Nevent):
            mtmp = np.append(mnew[iev * 4 : (iev + 1) * 4, ichain], mnew[-4:, ichain])
            dtmp = lf.get_times(m=mtmp, xr=xr, yr=yr, zr=zr, Nsta=Nsta)
            ist = iev * Ndatev
            iend = (iev + 1) * Ndatev
            dnew[ist:iend, ichain] = dtmp
            logLtmp[iev] = (
                -Ndatevp / 2.0 * np.log(mnew[-2, ichain])
                - np.sum((dobs[ist : ist + Nsta] - dtmp[0:Nsta]) ** 2)
                / (2.0 * mnew[-2, ichain] ** 2)
                - Ndatevs / 2.0 * np.log(mnew[-1, ichain])
                - np.sum((dobs[ist + Nsta : iend] - dtmp[Nsta:]) ** 2)
                / (2.0 * mnew[-1, ichain] ** 2)
            )

        logLkeep2[0, ichain] = np.sum(logLtmp)

    logLcur = logLkeep2[0, :].copy()
    dcur = dnew.copy()

    ########
    ######## LINROT
    ########
    # u2 = np.zeros((Npar,Npar,Nchain),dtype=float) #initialize u for SVD
    # pcsd2 = np.zeros((Npar,Nchain),dtype=float) #initialize pcsd for SVD
    if ilinrot:
        u[:, :, 0], pcsd[:, 0] = lf.linrot(
            m=mcur[:, 0],
            Nsta=Nsta,
            Ndat=Ndat,
            Ndatev=Ndatev,
            Npar=Npar,
            Nevent=Nevent,
            xr=xr,
            yr=yr,
            zr=zr,
            maxpert=maxpert,
        )
        u[:, :, 1], pcsd[:, 1] = lf.linrot(
            m=mcur[:, 1],
            Nsta=Nsta,
            Ndat=Ndat,
            Ndatev=Ndatev,
            Npar=Npar,
            Nevent=Nevent,
            xr=xr,
            yr=yr,
            zr=zr,
            maxpert=maxpert,
        )
    #    u2[:,:,0],pcsd2[:,0] = lf.linrot(m=mt,Nsta=Nsta,Npar=Npar,xr=xr,yr=yr,zr=zr,maxpert=maxpert)
    #    u2[:,:,1],pcsd2[:,1] = lf.linrot(m=mt,Nsta=Nsta,Npar=Npar,xr=xr,yr=yr,zr=zr,maxpert=maxpert)

    ###Counter / Toggle Variables
    ihead = True  # a toggle for writing the file (don't change this)
    iacc = np.zeros((Nchain, Npar))  # counter to calculate the acceptance rate
    iprop = np.zeros((Nchain, Npar))  # counter to calculate the acceptance rate
    ikeep = 0  # counter when writing to output files
    ncov = np.zeros(Nchain)  # initialize the dividing number for covarianve (in float)
    c_new = np.zeros(Nchain)  # toggle for covarianve on-and-off, don't change this
    icount = np.zeros(Nchain)  # counter for rotation matrix
    hist_d_plot = []  # to collect the hist_dif into a list
    irot_start = False
    irot_msg = True
    t1 = time.time()
    i_after_rot = 0
    for imcmc in np.arange(NMCMC):  # loop to number of MCMC random walk
        # t1=time.time()
        c_diff = np.max(np.max(np.abs(R[:, :, 0] - R[:, :, 1])))
        if (c_diff < cconv) & irot_msg & irot:  # when activating irot_start
            irot_start = True
            irot_msg = False
            print("Starting PC sampling with nonlinear estimate at imcmc=", imcmc)
        if irot_msg == False:
            i_after_rot += 1

        for ichain in np.arange(Nchain):  # loop to number of starting random points

            for ipar in np.arange(Npar):  # loop to number of parameter
                mtry = (mcur[:, ichain] - minlim) / maxpert
                mtry = np.matmul(np.transpose(u[:, :, ichain]), mtry)  # with rotation
                ## Cauchy proposal:

                mtry[ipar] = mtry[ipar] + 1.3 * pcsd[ipar, ichain] * np.tan(
                    np.pi * (np.random.rand(1)[0] - 0.5)
                )
                ## Gaussian proposal
                #            mtry[ipar] = mtry[ipar]+ 1.1*pcsd[ipar,ichain]*np.random.randn(1)
                mtry = np.matmul(u[:, :, ichain], mtry)
                mtry = minlim + (mtry * maxpert)

                iprop[ichain, ipar] += 1
                if ((maxlim - mtry).min() > 0.0) & (
                    (mtry - minlim).min() > 0.0
                ):  # Check that mtry is inside unifrom prior:

                    # calculate dtry from both Vp and Vs
                    # dtry = lf.get_times(m=mtry,xr=xr,yr=yr,zr=zr,Nsta=Nsta)
                    for iev in range(Nevent):
                        mtmp = np.append(mtry[iev * 4 : (iev + 1) * 4], mtry[-4:])
                        dtmp = lf.get_times(m=mtmp, xr=xr, yr=yr, zr=zr, Nsta=Nsta)
                        ist = iev * Ndatev
                        iend = (iev + 1) * Ndatev
                        dtry[ist:iend] = dtmp
                        logLtmp[iev] = (
                            -Ndatevp / 2.0 * np.log(mtry[-2])
                            - np.sum((dobs[ist : ist + Nsta] - dtmp[0:Nsta]) ** 2)
                            / (2.0 * mtry[-2] ** 2)
                            - Ndatevs / 2.0 * np.log(mtry[-1])
                            - np.sum((dobs[ist + Nsta : iend] - dtmp[Nsta:]) ** 2)
                            / (2.0 * mtry[-1] ** 2)
                        )
                    logLtry = np.sum(logLtmp)
                    #                logLtry = -Ndat/2.*np.log(mtry[-1])-np.sum((dobs-dtry)**2)/(2*mtry[-1]**2);

                    # Compute likelihood ratio in log space:
                    dlogL = logLtry - logLcur[ichain]
                    xi = np.random.rand(1)
                    # Apply MH criterion (accept/reject)
                    if xi <= np.exp(dlogL):
                        iacc[ichain, ipar] += 1
                        logLcur[ichain] = logLtry
                        mcur[:, ichain] = mtry
                        dcur[:, ichain] = dtry

            # t2=time.time()
            # print('time:',t2-t1)
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

            # print('Acc rate')
            # print(iacc[ichain,:]/iprop[ichain,:])
            # print(pcsd)
            mw = (mcur[:, ichain] - minlim) / maxpert  # for covariance
            ncov[ichain] += 1.0
            mmsum[:, ichain] = mmsum[:, ichain] + mw  # calculating the sum of mean(m)
            mbar[:, ichain] = (
                mmsum[:, ichain] / ncov[ichain]
            )  # calculating covariance matrix
            mcsum[:, :, ichain] = mcsum[:, :, ichain] + np.outer(
                np.transpose(mw - mbar[:, ichain]), mw - mbar[:, ichain]
            )
            if (c_new[ichain] == 2) & (icount[ichain] == 0):
                Cov[:, :, ichain] = (
                    mcsum[:, :, ichain] / ncov[ichain]
                )  # calculating covariance matrix
                for ipar in range(Npar):
                    for jpar in range(Npar):
                        R[ipar, jpar, ichain] = Cov[ipar, jpar, ichain] / np.sqrt(
                            Cov[ipar, ipar, ichain] * Cov[jpar, jpar, ichain]
                        )
                if irot_start & irot:  # ROTATION
                    u[:, :, ichain], s, vh = np.linalg.svd(
                        Cov[:, :, ichain]
                    )  # rotate it to its Singular Value Decomposition
                    pcsd[:, ichain] = np.sqrt(s)
            icount[ichain] += 1  # counter for rotation matrix

            if c_new[ichain] >= 1:  # calculate the difference between ichain parameters
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
        else:
            hist_d_plot = []  # to collect the hist_dif into a list

        if (iwrite) & (ikeep == NKEEP):  # dump to a file
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

            # print("saving output files %.1d %% "%imcmc)

            ihead = 0
            ikeep = 0

        elif ikeep == NKEEP:
            ihead = 0
            ikeep = 0
