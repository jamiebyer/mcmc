import numpy as np
from disba import PhaseDispersion
import pandas as pd
from scipy import interpolate


class VelocityModel:
    def __init__(self, thickness, vel_p, vel_s, density_params, sigma_pd):
        self.thickness = thickness
        self.vel_p = vel_p
        self.vel_s = vel_s
        self.sigma_pd = sigma_pd

        self.density = (vel_p - density_params[0]) / density_params[1]

        self.params = self.thickness  # and uncertainty?
        self.n_params = len(self.params)

        self.velocity_model = [self.thickness, self.vel_p, self.vel_s, self.density]
        self.vel_rayleigh = None

        # self.pcsd = 1 / 20
        # self.u = np.eye(n_params)

    def forward_model(freqs, velocity_model):
        """
        Get phase dispersion curve from shear velocities.
        """
        periods = 1 / freqs
        pd = PhaseDispersion(*velocity_model)
        pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
        # ell = Ellipticity(*velocity_model.T)

        return pd_rayleigh

    def get_density(vel_p, density_params):
        # Birch's law
        density = (vel_p - density_params[0]) / density_params[1]
        return density

    def get_birch_params():
        # fit to prem
        # density = (vel_p - density_params[0]) / density_params[1]
        prem = pd.read_csv("./PREM500_IDV.csv")

        radius = prem['radius[unit="m"]']
        prem_density = prem['density[unit="kg/m^3"]']

        # fit the curve
        density_params = np.polyfit(radius, prem_density, deg=1)
        # returns [-1.91018882e-03  1.46683536e+04]

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

    def generate_true_model(
        n_layers, layer_bounds, poisson_ratio, density_params, sigma_pd
    ):
        """
        Generate true model, which will be used to create simulated observed pd curves.
        """
        # from PREM...
        prem = pd.read_csv("./PREM500_IDV.csv")

        # generate layer thicknesses
        thickness = np.random.uniform(
            layer_bounds[0], layer_bounds[1], n_layers
        )  # change to gaussian
        # this is setting the depth to be the bottom of each layer ***
        depth = np.cumsum(thickness)
        radius = prem['radius[unit="m"]'] / 1000  # m -> km

        # interpolate density
        # prolly want the avg of each layer or something ***
        prem_density = prem['density[unit="kg/m^3"]'] / 1000  # kg/m^3 -> g/cm3
        density_func = interpolate.interp1d(radius, prem_density)
        density = density_func(depth)

        # get initial vs
        # velocities are split into components
        vsh = prem['Vsh[unit="m/s"]'] / 1000  # m/s -> km/s
        vsv = prem['Vsv[unit="m/s"]'] / 1000  # m/s -> km/s
        prem_vs = np.sqrt(vsh**2 + vsv**2)
        vs_func = interpolate.interp1d(radius, prem_vs)
        vel_s = vs_func(depth)

        # ***
        # get initial vp
        vp_vs = np.sqrt((2 - 2 * poisson_ratio) / (1 - 2 * poisson_ratio))
        vel_p = vel_s * vp_vs

        # or since it's the true model, use prem vp
        # vph = prem['Vph[unit="m/s"]'] / 1000  # m/s -> km/s
        # vpv = prem['Vpv[unit="m/s"]'] / 1000  # m/s -> km/s
        # prem_vp = np.sqrt(vph**2 + vpv**2)
        # vp_func = interpolate.interp1d(radius, prem_vp)
        # vp = vp_func(depth)

        # assemble velocity model
        velocity_model = VelocityModel(
            thickness, vel_p, vel_s, density_params, sigma_pd
        )

        return velocity_model

    def validate_params(params, bounds):
        # first bound applies to all layers
        # if that is different than n_params
        return np.all(params >= bounds[0, :]) and np.all(params <= bounds[1, :])

    def generate_starting_model(true_model, bounds, pcsd=0.05):
        """
        Loop until a valid starting model is created.
        """
        valid_model = False
        # maybe there's another way to generate starting model
        starting_model = true_model  # .copy()
        true_velocity_model = true_model.velocity_model.copy()  # ...

        while not valid_model:
            starting_model.velocity_model = (
                true_velocity_model + pcsd * np.random.randn(true_model.n_params)
            )
            # validate, check bounds
            valid_model = VelocityModel.validate_params(
                starting_model.velocity_model, bounds
            )

        # calculate likelihood
        starting_model.logL = VelocityModel.get_likelihood()

        return starting_model

    def perturb_model(self, bounds, layer_bounds, u, pcsd, scale_factor=1.3):

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
        for ind in range(n_params):
            # validate test params
            if not VelocityModel.validate_params(test_params[ind], bounds):
                continue

            # calculate new vel_s from new generated thicknesses

            # calculate new likelihood
            logL_new = VelocityModel.get_likelihood(
                pd_rayleigh, pd_rayleigh_obs, n_params, sigma_pd_rayleigh
            )

            # Compute likelihood ratio in log space:
            dlogL = logL_new - self.logL
            xi = np.random.rand(1)
            # Apply MH criterion (accept/reject)
            if xi <= np.exp(dlogL):
                self.logL = logLnew
                self.params = test_params  # validate this
                # self.vel_rayleigh =

    def get_jacobian(self, freqs, n_layers, dm_start, n_dm=50):
        """
        n_freqs is also n_depths | n_layers
        """
        n_freqs = len(freqs)
        dm = np.zeros(n_dm, self.n_params)
        dm[0] = dm_start
        dRdm = np.zeros((self.n_params, n_freqs, n_dm))

        # Estimate deriv for range of dm values
        for dm_ind in range(n_dm):
            model_pos = self.velocity_model.copy() + dm[dm_ind]
            model_neg = self.velocity_model.copy() - dm[dm_ind]

            for param_ind in range(self.n_params):
                model_pos[param_ind] = model_pos[param_ind] + dm[param_ind, dm_ind]

                # loop over each layer
                for layer_ind in range(n_layers):
                    pdr_pos = VelocityModel.forward_model(freqs, model_pos)
                    pdr_neg = VelocityModel.forward_model(freqs, model_neg)

                dRdm[param_ind, :, dm_ind] = np.where(
                    (np.abs((pdr_pos - pdr_neg) / (pdr_pos + pdr_neg)) > 1.0e-7 < 5),
                    (pdr_pos - pdr_neg) / (2.0 * dm[param_ind, dm_ind]),
                    dRdm[param_ind, :, dm_ind],
                )

                # setting dm for the next loop
                if dm_ind < n_dm - 1:
                    dm[:, dm_ind + 1] = dm[:, dm_ind] / 1.5

        Jac = VelocityModel.get_best_derivative(dRdm)

        return Jac

    def get_best_derivative(dRdm, n_layers, n_dm):
        for layer_ind in range(
            n_layers
        ):  # For each datum, choose best derivative estimate
            best = 1.0e10
            ibest = 1

            for dm_ind in range(n_dm - 2):
                # check if the derivative will very very large
                if np.any(np.abs(dRdm[ip, layer_ind, dm_ind : dm_ind + 2]) < 1.0e-7):
                    test = 1.0e20
                else:
                    test = np.abs(
                        np.sum(
                            (
                                dRdm[ip, layer_ind, dm_ind : dm_ind + 1]
                                / dRdm[ip, layer_ind, dm_ind + 1 : dm_ind + 2]
                            )
                            / 2
                            - 1
                        )
                    )

                if (test < best) and (test > 1.0e-7):
                    best = test
                    ibest = dm_ind + 1

            Jac[layer_ind, ip] = dRdm[ip, layer_ind, ibest]  # Best deriv into Jacobian
            if best > 1.0e10:
                Jac[layer_ind, ip] = 0.0

    def lin_rot(self, bounds, n_dm=50):

        Jac = self.get_jacobian()
        Jac[:, i] = Jac[:, i] * bounds[2, i]  # Scale columns of Jacobian for stability

        # what should this value be ??
        Mpriorinv = np.diag(self.n_params * [12])  # variance for U[0,1]=1/12

        Cdinv = no.diag(1 / sigma**2)  # what should sigma be ??

        JactCdinv = np.matmul(np.transpose(Jac), Cdinv)
        Ctmp = np.matmul(JactCdinv, Jac) + Mpriorinv

        V, L, VT = np.linalg.svd(Ctmp)
        pcsd = 0.5 * (1.0 / np.sqrt(np.abs(L)))  # PC standard deviations

        return pcsd

    def get_likelihood(vel_s, vel_s_obs, n_params, sigma_vel_s):
        """ """
        # probably using dispersion curves, compare directly to data
        residuals = vel_s_obs - vel_s

        logL = -(1 / 2) * n_params * np.log(sigma_vel_s) - np.sum(res_p**2) / (
            2 * sigma_vel_s**2
        )

        return np.sum(logL)
