from wave_functions import phase_dispersion
from numpy import np


class VelocityModel:
    def __init__(self, vel_p, vel_s, sigma_p, sigma_s, thicknesses):
        """
        vel_p: P wave velocities.
        vel_s: S wave velocities.
        sigma_p: P wave velocity uncertainties.
        sigma_s: S wave velocity uncertainties.
        thicknesses: Earth layer thicknesses.

        """
        self.vel_p = vel_p
        self.vel_s = vel_s
        self.sigma_p = sigma_p
        self.sigma_s = sigma_s
        self.thicknesses = thicknesses

    def calculate_vel_s(self, t):
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

        pd = PhaseDispersion(*velocity_model.T)
        pd_rayleigh = pd(t, mode=0, wave="rayleigh")

        # ell = Ellipticity(*velocity_model.T)

        freq_rayleigh = pd_rayleigh.velocity
        vel_rayleigh = pd_rayleigh.velocity

        # plot vs. freq, wavelength, depth
        # plot of velocity model
        # plot ellipticity

        # get vel_s

        return vel_s
