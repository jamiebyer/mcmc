import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
from numba.core.errors import TypingError
import xarray as xr

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

    def __init__(self, n_layers, param_bounds, proposal_width, vpvs_ratio):
        # initialize params
        self.n_layers = n_layers
        self.vpvs_ratio = vpvs_ratio
        self.param_bounds = param_bounds
        self.proposal_width = proposal_width

        # get number of parameters
        self.n_model_params = (2 * self.n_layers) + 1
        self.model_params = np.empty(self.n_model_params)

        # model parameter inds
        self.depth_inds = np.full(self.n_model_params, False)
        self.depth_inds[np.arange(self.n_layers)] = True

        self.vel_s_inds = np.full(self.n_model_params, False)
        self.vel_s_inds[np.arange(self.n_layers, 2 * self.n_layers + 1)] = True

        # used by inversion to define dataset for storing parameters
        # for defining the dataset, need the names and size of the params
        # and with inds
        self.params_info = {
            "depth": {"n_params": n_layers, "inds": self.depth_inds, "units": "km"},
            "vel_s": {
                "n_params": n_layers + 1,
                "inds": self.vel_s_inds,
                "units": "km/s",
            },
        }

    def get_model_params_dict(self):
        model_params_dict = {
            "coords": {
                "n_model_params": {
                    "dims": ["n_model_params"],
                    "data": np.arange(self.n_model_params),
                },
                "n_bounds": {
                    "dims": ["n_bounds"],
                    "data": np.arange(3),
                },
            },
            "data_vars": {
                "param_bounds": {
                    "dims": ["n_model_params", "n_bounds"],
                    "data": self.assemble_param_bounds(),
                }
            },
            "attrs": {
                "n_layers": self.n_layers,
                "vpvs_ratio": self.vpvs_ratio,
                # "depth_bounds": self.param_bounds["depth"],
                # "vel_s_bounds": self.param_bounds["vel_s"],
                "depth_proposal_width": self.proposal_width["depth"],
                "vel_s_proposal_width": self.proposal_width["vel_s"],
            },
        }

        # dimension could be n_model_params
        # change in def
        for key, val in self.params_info.items():
            model_params_dict["data_vars"].update(
                {key + "_inds": {"dims": ["n_model_params"], "data": val["inds"]}}
            )

        return model_params_dict

    # functions to compute nuissance params from model params
    def get_vel_p(self, vel_s):
        vel_p = vel_s * self.vpvs_ratio
        return vel_p

    def get_density(self, vel_p):
        # using Garner's relation
        density = (1741 * np.sign(vel_p) * abs(vel_p) ** (1 / 4)) / 1000
        return density

    def assemble_param_bounds(self):
        # reshape bounds to be the same shape as params
        depth_bounds = self.param_bounds["depth"]
        vel_s_bounds = self.param_bounds["vel_s"]

        if len(depth_bounds.shape) == 1:
            depth_bounds = depth_bounds * self.n_layers
        if len(vel_s_bounds.shape) == 1:
            vel_s_bounds = vel_s_bounds * (self.n_layers + 1)

        param_bounds = np.concatenate(
            (
                depth_bounds,
                vel_s_bounds,
            ),
            axis=0,
        )

        # add the range of the bounds to param_bounds as a third column (min, max, range)
        range = param_bounds[:, 1] - param_bounds[:, 0]
        param_bounds = np.column_stack((param_bounds, range))

        return param_bounds

    def assemble_proposal_width(self):
        """
        from -- for each param.
        """
        # reshape bounds to be the same shape as params
        depth_width = self.proposal_width["depth"]
        vel_s_width = self.proposal_width["vel_s"]

        if len(depth_width) == 1:
            depth_width = depth_width * self.n_layers
        if len(vel_s_width) == 1:
            vel_s_width = vel_s_width * (self.n_layers + 1)

        proposal_width = np.concatenate(
            (
                depth_bounds,
                vel_s_bounds,
            ),
            axis=0,
        )

        return proposal_width

    def validate_physics(self, model_params):
        # validate parameter bounds
        # vel_s = model_params[self.vel_s_inds]
        # return np.all(vel_s[-1] > vel_s[:-1])
        return True

    def sort_layers(self, model_params):
        """"""
        # generated params are depths of interfaces
        # sort depths and corresponding params (vel_s)
        # one of the layers will never move
        # define depths with interface at z=0

        # sort layers
        depth = model_params[self.depth_inds]
        vel_s = model_params[self.vel_s_inds]

        # depth = np.concatenate(([0], depth))
        inds = np.argsort(depth)
        depth = depth[inds]
        vel_s[1:] = vel_s[1:][inds]

        model_params[self.depth_inds] = depth
        model_params[self.vel_s_inds] = vel_s

        return model_params

    def forward_model(self, periods, model_params):
        """
        get phase dispersion curve for current shear velocities and layer thicknesses.

        :param periods:
        :param velocity model: velocity model for disba has the format
            [thickness (km), vel_p (km/s), vel_s (km/s), density (g/cm3)]
        :param model_params: model params to use to get phase dispersion
        """
        depth = model_params[self.depth_inds]
        vel_s = model_params[self.vel_s_inds]
        # get thicknesses
        depth = np.concatenate(([0], depth))
        thickness = np.concatenate((depth[1:] - depth[:-1], [0]))

        # assemble params into velocity model
        vel_p = self.get_vel_p(vel_s)
        density = self.get_density(vel_p)
        # avoid converting thickness back and forth from list
        velocity_model = np.array([thickness, vel_p, vel_s, density])

        # phase dispersion object
        pd = PhaseDispersion(*velocity_model)

        # try calculating phase_velocity from given params.
        try:
            pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
            phase_velocity = pd_rayleigh.velocity

            return phase_velocity
        except (DispersionError, ZeroDivisionError, TypingError) as e:
            # *** track the type of error ***
            # failed to find root for fundamental mode
            # division by zero
            # raise e
            raise e


class EllipticityParams(ModelParams):
    pass
