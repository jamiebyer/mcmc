import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

np.complex_ = np.complex64


class ModelParams:
    def __init__(
        self,
        values,
    ):
        pass

    def forward_problem(forward_problem_func, params):
        params = assemble_params(params)
        forward_problem_func(*params)


class DispersionCurveParams(ModelParams):

    def __init__(
        self,
    ):

        self.n_layers = n_layers

        self.vpvs_ratio = vpvs_ratio
        self.model_params = np.empty(self.n_params)

    def density():
        pass

    def dispersion_curve_forward_problem():

        pass

    @property
    def layer_bounds(self):
        return self.param_bounds[0]

    @property
    def thickness(self):
        return self.model_params[: (self.n_layers - 1)]
        # return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        # validate bounds
        self.model_params[: (self.n_layers - 1)] = thickness
        # self._thickness = thickness

    @property
    def vel_s(self):
        return self.model_params[self.n_layers : 2 * self.n_layers]
        # return self._vel_s

    @vel_s.setter
    def vel_s(self, vel_s):
        # validate bounds
        self.model_params[self.n_layers : 2 * self.n_layers] = vel_s
        # self._vel_s = vel_s

        # update vel_p and density
        self.vel_p = self.get_vel_p(self._vel_s)
        self.density = self.get_density(self.vel_p)

    @property
    def vel_p(self):
        # check bounds
        return self.model_params[self.n_layers : 2 * self.n_layers]
        # return self._vel_p

    @vel_p.setter
    def vel_p(self, vel_p):
        # check bounds
        self.model_params[self.n_layers : 2 * self.n_layers] = vel_p
        # self._vel_p = vel_p

    @property
    def density(self):
        # check bounds
        return self.model_params[self.n_layers : 2 * self.n_layers]
        # return self._density

    @density.setter
    def density(self, density):
        # check bounds
        self.model_params[self.n_layers : 2 * self.n_layers] = density
        # self._density = density

    def get_vel_p(self, vel_s):
        vel_p = vel_s * self.vpvs_ratio
        return vel_p

    def get_density(self, vel_p):
        # using Garner's relation
        density = (1741 * np.sign(vel_p) * abs(vel_p) ** (1 / 4)) / 1000
        return density

    @staticmethod
    def assemble_param_bounds(bounds, n_layers):
        # reshape bounds to be the same shape as params
        param_bounds = np.concatenate(
            (
                [bounds["thickness"]] * (n_layers - 1),
                [bounds["vel_s"]] * n_layers,
                [bounds["vel_p"]] * n_layers,
                [bounds["density"]] * n_layers,
                # [bounds["sigma_model"]],
            ),
            axis=0,
        )

        # add the range of the bounds to param_bounds as a third column (min, max, range)
        range = param_bounds[:, 1] - param_bounds[:, 0]
        param_bounds = np.column_stack((param_bounds, range))

        return param_bounds

    def validate_bounds():
        # validate bounds and physics...
        valid_thickness = (thickness >= param_bounds["thickness"][0]) & (
            thickness <= param_bounds["thickness"][1]
        )
        valid_vel_s = (vel_s >= param_bounds["vel_s"][0]) & (
            vel_s <= param_bounds["vel_s"][1]
        )
        """
        valid_vel_p = (vel_p >= param_bounds["vel_p"][0]) & (
            vel_p <= param_bounds["vel_p"][1]
        )
        valid_density = (density >= param_bounds["density"][0]) & (
            density <= param_bounds["density"][1]
        )
        """
        valid_params = np.all(
            valid_thickness & valid_vel_s  # & valid_vel_p & valid_density
        )

    def get_velocity_model(self, param_bounds, thickness, vel_s):
        vel_p = self.get_vel_p(vel_s)
        density = self.get_density(vel_p)
        velocity_model = np.array([list(thickness) + [0], vel_p, vel_s, density])

        return velocity_model, valid_params

    def forward_model(self, periods, velocity_model):
        """
        get phase dispersion curve for current shear velocities and layer thicknesses.

        :param model_params: model params to use to get phase dispersion
        *** generalize later***
        """
        # *** keep track of errors in forward model
        # get phase dispersion curve
        # thickness, Vp, Vs, density
        # km, km/s, km/s, g/cm3
        pd = PhaseDispersion(*velocity_model)

        # try calculating phase_velocity from given params.
        try:
            pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
            # ell = Ellipticity(*velocity_model.T)
            phase_velocity = pd_rayleigh.velocity

            return phase_velocity
        except (DispersionError, ZeroDivisionError) as e:
            # *** errors: ***
            # failed to find root for fundamental mode
            # division by zero
            raise e


class EllipticityParams(ModelParams):
    pass
