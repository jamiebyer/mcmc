import numpy as np
from disba import PhaseDispersion
import pandas as pd
from scipy import interpolate


class VelocityModel:
    def __init__(self, thickness, vel_p, vel_s, sigma_rayleigh, density_params):
        self.thickness = thickness
        self.vel_p = vel_p
        self.vel_s = vel_s
        self.sigma_rayleigh = sigma_rayleigh

        self.density = (vel_p - density_params[0]) / density_params[1]

        self.velocity_model = [self.thickness, self.vel_p, self.vel_s, self.density]
        # self.vel_rayleigh = None

        # self.pcsd = 1 / 20
        # self.u = np.eye(n_params)

    def forward_model(self):
        """
        Get phase dispersion curve from shear velocities.
        """
        pd = PhaseDispersion(*self.velocity_model)
        pd_rayleigh = pd(self.periods, mode=0, wave="rayleigh")
        # ell = Ellipticity(*velocity_model.T)

        return pd_rayleigh

    def get_density(vel_p, density_params):
        # Birch's law
        density = (vel_p - density_params[0]) / density_params[1]
        return density

    def get_vel_s_profile(self, freq, vel_rayleigh):
        # get wavelength from frequency
        wavelength_rayleigh = vel_rayleigh / freq  # CHECK UNITS

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

    def generate_true_model(n_layers, layer_bounds, poisson_ratio):
        # from PREM...
        prem = pd.read_csv("./PREM500_IDV.csv")

        # generate layer thicknesses
        thickness = np.random.uniform(layer_bounds[0], layer_bounds[1], n_layers) # change to gaussian
        depth = np.cumsum(thickness)
        radius = prem["radius"] / 1000  # convert m to km

        # interpolate density
        # prolly want the avg of each layer or something
        prem_density = prem["density"]  # kg/m^3
        density_func = interpolate.interp1d(radius, prem_density)
        density = density_func(depth)

        # get initial vs
        prem_vs = prem["Vsh"]/1000  # convert m/s to km/s
        vs_func = interpolate.interp1d(radius, prem_vs)
        vs = vs_func(depth)

        # get initial vp
        vp_vs = np.sqrt((2-2*poisson_ratio)/(1-2*poisson_ratio))
        vp = vs * vp_vs

        # or since it's the true model, use prem vp

        # assemble velocity model
        velocity_model = [thickness, vp, vs, density]
        return velocity_model

    def validate_params(params, bounds):
        return np.all(params >= bounds[0, :]) and np.all(params <= bounds[1, :])


    def generate_starting_model(true_model, n_params, pcsd=0.05):
        # maybe there's another way to generate starting model
        starting_model = true_model.copy().velocity_model + pcsd * np.random.randn(
            n_params
        )

        # validate, check bounds
        # calculate likelihood
        starting_model.update_likelihood()
        return starting_model


    def perturb_model(self, layer_bounds, u, pcsd, scale_factor=1.3):
        """
        Solving for:
        - thickness of each layer
        - vs of each layer
        - vp/vs ?
        - sigma s and sigma p ?
        """

        # self is current best model
        # current_params = self.params
        thickness = self.thickness
        n_params = len(thickness)  # unless there are uncertainties too

        # normalizing, rotating
        test_params = (thickness - layer_bounds[0]) / layer_bounds[2]
        test_params = np.matmul(np.transpose(u), test_params)

        # generate params to try; Cauchy proposal
        test_params += (
            scale_factor
            * self.pcsd
            * np.tan(np.pi * (np.random.rand(len(test_params)) - 0.5))
        )

        # rotating back and scaling again
        test_params = np.matmul(u, test_params)
        test_params = layer_bounds[0] + (test_params * layer_bounds[2])

        # perturb each parameter in the model

        # loop over params and perturb
        # loop over each layer

        for ind in range(n_params):
            # validate test params
            # calculate new likelihood
            logL_new = self.update_likelihood(param_ind=ind, param=test_params[ind])
            # check likelihood with each observed data
            # model likelihood is sum of each likelihood from data

            logLtry = np.sum(logL)

            # Compute likelihood ratio in log space:
            dlogL = logLtry - logLcur
            xi = np.random.rand(1)
            # Apply MH criterion (accept/reject)
            if xi <= np.exp(dlogL):
                logLcur = logLtry
                mcur = mtry
                dcur = dtry


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

    def lin_rot(self, n_dm=50, n_params):
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

    def update_likelihood(self, vel_s, vel_s_obs, n_params, sigma_vel_s):
        """ """
        residuals = vel_s_obs - vel_s

        logL = -(1 / 2) * n_params * np.log(sigma_vel_s) -np.sum(res_p**2) / (2 * sigma_vel_s**2)

        return np.sum(logL)