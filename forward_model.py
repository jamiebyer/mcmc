import numpy as np
from disba import PhaseDispersion


class ForwardModel:
    def __init__(self, vel_p, vel_s, sigma_p, sigma_s, velocity_model):
        """
        vel_p: P wave velocities.
        vel_s: S wave velocities.
        sigma_p: P wave velocity uncertainties.
        sigma_s: S wave velocity uncertainties.
        velocity_model: [[thickness (km), Vp (km/s), Vs(km/s), density(g/cm3)]]

        """
        self.vel_p = vel_p
        self.vel_s = vel_s
        self.sigma_p = sigma_p
        self.sigma_s = sigma_s
        self.velocity_model = velocity_model
        self.covariance_matrix = None
        self.params = [self.vel_p, self.vel_s, self.sigma_p, self.sigma_s]
        self.n_params = len(self.params)
        self.logL = None

    def generate_starting_model(self, prior_model, bounds, pcsd, periods):
        prior_params = prior_model.params
        n_params = len(prior_params)
        new_params = []
        valid_model = False
        while not valid_model:
            new_params = prior_params + pcsd * np.random.randn(n_params)
            valid_model = True
            # check bounds
            for ind in range(n_params):
                if (bounds[ind][0] > new_params[ind]) or (
                    bounds[ind][1] < new_params[ind]
                ):
                    valid_model = False
                    break

            # pd_rayleigh constraint
            pd_rayleigh = self.get_rayleigh_phase_dispersion(periods)
            if np.min(pd_rayleigh) < self.vel_s:
                valid_model = False

        self.vel_p, self.vel_s, self.sigma_p, self.sigma_s = new_params

    def generate_perturbed_model(self, u, pcsd):
        ## Cauchy proposal:
        """
        mtry[ipar] = mtry[ipar] + 1.3 * pcsd[ipar, ichain] * np.tan(
            np.pi * (np.random.rand(1)[0] - 0.5)
        )
        ## Gaussian proposal
        #            mtry[ipar] = mtry[ipar]+ 1.1*pcsd[ipar,ichain]*np.random.randn(1)
        mtry = np.matmul(u[:, :, ichain], mtry)
        # unnormalize
        mtry = minlim + (mtry * maxpert)
        """
        # check bouncs

        # return model

        return None

    def perturb_params(self):

        for param in len(self.params):
            # generate model
            model = self.generate_model()
            if mtry is not None:
                # calculate dtry from both Vp and Vs
                # dtry = lf.get_times(m=mtry,xr=xr,yr=yr,zr=zr,Nsta=Nsta)
                logL = mtry.calculate_likelihood(events)

                logLtry = np.sum(logL)

                # Compute likelihood ratio in log space:
                dlogL = logLtry - logLcur[ichain]
                xi = np.random.rand(1)
                # Apply MH criterion (accept/reject)
                if xi <= np.exp(dlogL):
                    logLcur[ichain] = logLtry
                    mcur[:, ichain] = mtry
                    dcur[:, ichain] = dtry

    def get_rayleigh_phase_dispersion(self, t, mode=0):
        pd = PhaseDispersion(*self.velocity_model.T)
        pd_rayleigh = pd(t, mode=mode, wave="rayleigh")
        # ell = Ellipticity(*velocity_model.T)
        return pd_rayleigh

    def get_vel_s_profile(self, t):
        """
        t: array of periods
        """
        # make velocity model
        """
        # thickness, Vp, Vs, density
        # km, km/s, km/s, g/cm3
        velocity_model = np.array(
                [
                    [10.0, 7.00, 3.50, 2.00],
                    [10.0, 6.80, 3.40, 2.00],
                    [10.0, 7.00, 3.50, 2.00],
                    [10.0, 7.60, 3.80, 2.00],
                    [10.0, 8.40, 4.20, 2.00],
                    [10.0, 9.00, 4.50, 2.00],
                    [10.0, 9.40, 4.70, 2.00],
                    [10.0, 9.60, 4.80, 2.00],
                    [10.0, 9.50, 4.75, 2.00],
                ]
            )
        """

        pd_rayleigh = self.get_rayleigh_phase_dispersion(t)

        freq_rayleigh = pd_rayleigh.velocity
        vel_rayleigh = pd_rayleigh.velocity

        # get wavelength from frequency
        wavelength_rayleigh = vel_rayleigh / freq_rayleigh  # CHECK UNITS

        # get depth from wavelength
        depths = wavelength_rayleigh

        # get vel_s depth profile
        vel_s = [vel_rayleigh[0]]
        vel_s_avgs = [vel_rayleigh[0]]
        total_avg = vel_rayleigh[0]
        for i in range(1, len(vel_rayleigh)):
            new_vel = (vel_rayleigh[i] - total_avg) / depths[
                i
            ]  # CHECK IF DEPTHS ARE ORDERED
            vel_s.append(new_vel)
            total_avg += new_vel

        return depths, vel_s

    def lin_rot():
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

    def get_derivatives(n_dm=50):
        dm = np.zeros()
        dm[0] = dm_start[ip]  # Estimate deriv for range of dm values
        for i in range(NDM):
            # print i,ip
            mtry1 = m.copy()
            mtry1[ip] = mtry1[ip] + dm[i]
            # dat1 = get_times(m=mtry1,xr=xr,yr=yr,zr=zr,Nsta=Nsta)
            for iev in range(Nevent):
                mtmp = np.append(mtry1[iev * 4 : (iev + 1) * 4], mtry1[-4:])
                dtmp = get_times(m=mtmp, xr=xr, yr=yr, zr=zr, Nsta=Nsta)
                ist = iev * Ndatev
                iend = (iev + 1) * Ndatev
                dat1[ist:iend] = dtmp

            mtry2 = m.copy()
            mtry2[ip] = mtry2[ip] - dm[i]
            # dat2 = get_times(m=mtry2,xr=xr,yr=yr,zr=zr,Nsta=Nsta)
            for iev in range(Nevent):
                mtmp = np.append(mtry2[iev * 4 : (iev + 1) * 4], mtry2[-4:])
                dtmp = get_times(m=mtmp, xr=xr, yr=yr, zr=zr, Nsta=Nsta)
                ist = iev * Ndatev
                iend = (iev + 1) * Ndatev
                dat2[ist:iend] = dtmp

            for j in range(ntot):
                if np.abs((dat1[j] - dat2[j]) / (dat1[j] + dat2[j])) > 1.0e-7:
                    dRdm[ip, j, i] = (dat1[j] - dat2[j]) / (2.0 * dm[i])
                else:
                    dRdm[ip, j, i] = 0.0
            if i < NDM - 1:
                dm[i + 1] = dm[i] / 1.5

    def get_best_derivative():
        for j in range(ntot):  # For each datum, choose best derivative estimate
            best = 1.0e10
            ibest = 1
            for i in range(NDM - 2):
                if (
                    (np.abs(dRdm[ip, j, i + 0]) < 1.0e-7)
                    or (np.abs(dRdm[ip, j, i + 1]) < 1.0e-7)
                    or (np.abs(dRdm[ip, j, i + 2]) < 1.0e-7)
                ):
                    test = 1.0e20
                else:
                    test = np.abs(
                        (
                            dRdm[ip, j, i + 0] / dRdm[ip, j, i + 1]
                            + dRdm[ip, j, i + 1] / dRdm[ip, j, i + 2]
                        )
                        / 2.0
                        - 1.0
                    )

                if (test < best) and (test > 1.0e-7):
                    best = test
                    ibest = i + 1
            Jac[j, ip] = dRdm[ip, j, ibest]  # Best deriv into Jacobian
            if best > 1.0e10:
                Jac[j, ip] = 0.0

    def lin_rot(self, n_dm=50):
        """
        ntot = Ndat
        dat1 = np.zeros(ntot)
        dat2 = np.zeros(ntot)
        dRdm = np.zeros((Npar, ntot, n_dm))
        dm_start = np.zeros(Npar)
        Jac = np.zeros((ntot, Npar))
        JactCdinv = np.zeros((ntot, ntot))
        Cdinv = np.zeros((ntot, ntot))
        Mpriorinv = np.zeros((Npar, Npar))
        Ctmp = np.zeros((Npar, Npar))
        VT = np.zeros((Npar, Npar))
        V = np.zeros((Npar, Npar))
        L = np.zeros(Npar)
        pcsd = np.zeros(Npar)
        """

        dm_start = m * 0.1
        sigma = m[-2:]

        for param in params:
            # calculate n_loops derivatives
            self.get_derivatives()
            self.get_best_derivative

        for i in range(Npar):  # Scale columns of Jacobian for stability
            Jac[:, i] = Jac[:, i] * maxpert[i]

        for i in range(Npar):
            Mpriorinv[i, i] = 12.0  # variance for U[0,1]=1/12

        i = 0
        for iev in range(Nevent):
            for ifq in range(2):
                for ista in range(Nsta):
                    Cdinv[i, i] = 1.0 / sigma[ifq] ** 2
                    i += 1
        JactCdinv = np.matmul(np.transpose(Jac), Cdinv)
        Ctmp = np.matmul(JactCdinv, Jac) + Mpriorinv
        # Ctmp = np.linalg.inv(np.matmul(JactCdinv,Jac) + Mpriorinv)

        V, L, VT = np.linalg.svd(Ctmp)
        pcsd = 0.5 * (1.0 / np.sqrt(np.abs(L)))  # PC standard deviations
        # pcsd = np.sqrt(L) # PC standard deviations

    def update_likelihood(self, station_positions, events, param_ind=None, param=None):
        if param is not None:
            params_og = self.params[param_ind]
            logL_og = self.logL
            self.params[param_ind] = param

        logL_new = 0
        for event in events:
            # calculate log(L) for each event
            t_obs_p = event.t_obs_p
            t_obs_s = event.t_obs_s

            t_model_p, t_model_s = event.get_times(station_positions, self)

            res_p = t_obs_p - t_model_p  # calculate p wave residual
            res_s = t_obs_s - t_model_s  # calculate s wave residual

            n_data_p = len(res_p)
            n_data_s = len(res_s)

            logL_p = -(1 / 2) * n_data_p * np.log(self.sigma_p)
            -np.sum(res_p**2) / (2 * self.sigma_p**2)

            logL_s = -(1 / 2) * n_data_s * np.log(self.sigma_s)
            -np.sum(res_s**2) / (2 * self.sigma_s**2)

            logL_new += logL_p + logL_s  # check dimensions

        if param is not None and logL_og > logL_new:  ##### CHECK
            self.params = params_og
        else:
            self.logL = logL_new

    def perturb_params(self, station_positions, events, scale_factor=1.3):
        # perturb each parameter in the model
        params = self.params
        for ind in len(params):
            # new perturbed model param

            # generate model
            self.update_likelihood(
                station_positions, events, param_ind=ind, param=params[ind]
            )

            ## Cauchy proposal:
            mtry[ipar] = mtry[ipar] + 1.3 * pcsd[ipar, ichain] * np.tan(
                np.pi * (np.random.rand(1)[0] - 0.5)
            )
            ## Gaussian proposal
            #            mtry[ipar] = mtry[ipar]+ 1.1*pcsd[ipar,ichain]*np.random.randn(1)
            mtry = np.matmul(u[:, :, ichain], mtry)
            mtry = minlim + (mtry * maxpert)

            if ((maxlim - mtry).min() > 0) and (
                (mtry - minlim).min() > 0
            ):  # Check that mtry is inside unifrom prior:

                # calculate dtry from both Vp and Vs
                # dtry = lf.get_times(m=mtry,xr=xr,yr=yr,zr=zr,Nsta=Nsta)
                # ... mtry.calculate_likelihood
                logL = self.calculate_likelihood(events)

                logLtry = np.sum(logL)

                # Compute likelihood ratio in log space:
                dlogL = logLtry - logLcur[ichain]
                xi = np.random.rand(1)
                # Apply MH criterion (accept/reject)
                if xi <= np.exp(dlogL):
                    logLcur[ichain] = logLtry
                    mcur[:, ichain] = mtry
                    dcur[:, ichain] = dtry
