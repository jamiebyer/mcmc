import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../mcmc")

import xarray as xr
import numpy as np

from run import run

"""
DEFAULT VALUES FOR RUNNING INVERSION

true_model_kwargs = (
    {
        "poisson_ratio": 0.265,
        "density_params": [540.6, 360.1],  # *** check units ***
        "n_data": 10,
        "n_layers": 10,
        "layer_bounds": [5e-3, 15e-3],  # km
        "vel_s_bounds": [2, 6],  # km/s
        "sigma_pd_bounds": [0, 1],
    },
)
inversion_kwargs = (
    {
        "poisson_ratio": 0.265,
        "density_params": [540.6, 360.1],  # *** check units ***
        "n_layers": 10,
        "n_chains": 2,
        "beta_spacing_factor": 1.15,
        "model_variance": 12,
        "n_bins": 200,
        "n_burn": 10000,
        "n_keep": 2000,
        "n_rot": 40000,
    },
)
run(
    true_model_kwargs,
    inversion_kwargs,
    n_data=10,
    max_perturbations=10,
    hist_conv=0.05,
    out_dir="/out/inversion_results",
)
"""


def create_test_file():
    # run inversion to the end of the burn-in and save results to file
    true_model_kwargs = (
        {},
    )
    inversion_kwargs = (
        {
            "n_burn": 2400,
            "n_keep": 200,
            "n_rot": 0,
        },
    )
    run(
        true_model_kwargs,
        inversion_kwargs,
        out_dir="/test_inversion_results",
    )


def test_generating_model_params():
    # check generating models, what parameters are valid, in bounds
    pass


def test_lin_rot():
    pass


def test_get_betas():
    betas = [0.5, 1]  # *** hard coded beta for now ***
    pass


def test_parallel_computing():
    pass


def test_rotation():
    # save values of running to the end of burn-in, use those to test rotation
    pass

def test_convergence():
    # *** test values:
    model_params[self.n_layers : 2 * self.n_layers] = [
        3.50,
        3.40,
        3.50,
        3.80,
        4.20,
        4.50,
        4.70,
        4.80,
        4.75,
        4.75,
    ]

create_test_file()