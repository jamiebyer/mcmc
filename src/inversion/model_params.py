import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError

np.complex_ = np.complex64


class ModelParams:

    def __init__(
        self,
    ):
        """
        model parameterization.
        class to hold specific model parameters to generalize Model.
        """
        pass

    def forward_problem():
        pass


class DispersionCurveParams(ModelParams):

    def __init__(self, n_layers, param_bounds, sigma_model, vpvs_ratio):
        # initialize params
        self.n_layers = n_layers
        self.vpvs_ratio = vpvs_ratio
        self.param_bounds = param_bounds
        self.sigma_model = sigma_model

        # get number of parameters
        self.n_model_params = (2 * self.n_layers) + 1
        self.n_nuissance_params = 2 * self.n_layers

        self.model_params = np.empty(self.n_model_params)
        self.nuissance_params = np.empty(self.n_nuissance_params)

        # model parameter inds
        self.thickness_inds = np.arange(self.n_layers)
        self.vel_s_inds = np.arange(self.n_layers, 2 * self.n_layers + 1)
        # nuissance parameter inds
        self.vel_p_inds = np.arange(self.n_nuissance_params)
        self.density_inds = np.arange(
            self.n_nuissance_params, 2 * self.n_nuissance_params
        )

        # used by inversion to define dataset for storing parameters
        # for defining the dataset, need the names and size of the params
        # and with inds
        # with this, are the inds needed anywhere else?
        self.params_info = {
            "thickness": {
                "n_params": n_layers,
                "inds": self.thickness_inds,
            },
            "vel_s": {
                "n_params": n_layers + 1,
                "inds": self.vel_s_inds,
            },
        }

    # functions to compute nuissance params from model params
    def get_vel_p(self, vel_s):
        vel_p = vel_s * self.vpvs_ratio
        vel_p = np.array([1.6, 2.5])
        return vel_p

    def get_density(self, vel_p):
        # using Garner's relation
        density = (1741 * np.sign(vel_p) * abs(vel_p) ** (1 / 4)) / 1000
        density = np.array([2.0, 2.5])
        return density

    def assemble_param_bounds(self):
        # reshape bounds to be the same shape as params
        param_bounds = np.concatenate(
            (
                [self.param_bounds["thickness"]] * self.n_layers,
                [self.param_bounds["vel_s"]] * (self.n_layers + 1),
            ),
            axis=0,
        )

        # add the range of the bounds to param_bounds as a third column (min, max, range)
        range = param_bounds[:, 1] - param_bounds[:, 0]
        param_bounds = np.column_stack((param_bounds, range))

        return param_bounds

    def assemble_sigma_model(self):
        """
        from sigma for each param.
        """
        sigma_model = np.concatenate(
            (
                [self.sigma_model["thickness"]] * self.n_layers,
                [self.sigma_model["vel_s"]] * (self.n_layers + 1),
            ),
            axis=0,
        )

        return sigma_model

    def get_velocity_model(self, model_params):
        """
        assemble params into the format of velocity model used by the forward model.
        needs to work for getting velocity model for test parameters.
        not static because get_vel_p requires vpvs_ratio.
        """
        thickness = model_params[self.thickness_inds]
        vel_s = model_params[self.vel_s_inds]
        vel_p = self.get_vel_p(vel_s)
        density = self.get_density(vel_p)

        # *** pre-allocate space here ***
        velocity_model = np.array([list(thickness) + [0], vel_p, vel_s, density])

        return velocity_model

    def forward_model(self, periods, velocity_model):
        """
        get phase dispersion curve for current shear velocities and layer thicknesses.

        :param periods:
        :param velocity model: velocity model for disba has the format
            [thickness (km), vel_p (km/s), vel_s (km/s), density (g/cm3)]
        :param model_params: model params to use to get phase dispersion

        """
        # assemble params into velocity model

        pd = PhaseDispersion(*velocity_model)

        # try calculating phase_velocity from given params.
        try:
            pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
            phase_velocity = pd_rayleigh.velocity

            return phase_velocity
        except (DispersionError, ZeroDivisionError) as e:
            # *** track the type of error ***
            # failed to find root for fundamental mode
            # division by zero
            raise e


class EllipticityParams(ModelParams):
    pass
