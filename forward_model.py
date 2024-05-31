from numpy import np
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

    def get_rayleigh_phase_dispersion(self, t, mode=0):
        pd = PhaseDispersion(*self.velocity_model.T)
        pd_rayleigh = pd(t, mode=mode, wave="rayleigh")
        # ell = Ellipticity(*velocity_model.T)
        return pd_rayleigh

    def get_vel_s_bounds(self, t):
        pd_rayleigh = self.get_rayleigh_phase_dispersion(t)

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

        return vel_s
