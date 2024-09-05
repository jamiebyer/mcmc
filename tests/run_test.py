# *** can this be in init?
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

import xarray as xr
import numpy as np

from run import run, setup_scene
from inversion import Inversion

import asyncio

"""
DEFAULT VALUES FOR RUNNING INVERSION

true_model_kwargs = {
    "poisson_ratio": 0.265,
    "density_params": [540.6, 360.1],  # *** check units ***
    "n_data": 10,
    "n_layers": 10,
    "layer_bounds": [5e-3, 15e-3],  # km
    "vel_s_bounds": [2, 6],  # km/s
    "sigma_pd_bounds": [0, 1],
}
inversion_kwargs = {
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
}
run(
    true_model_kwargs,
    inversion_kwargs,
    n_data=10,
    max_perturbations=10,
    hist_conv=0.05,
    out_dir="/out/inversion_results",
)
"""

def setup_inversion_obj():
    inversion = run(out_dir="/test_inversion_results")
    return inversion



def create_test_file():
    # run inversion to the end of the burn-in and save results to file
    inversion_kwargs = {
        "n_burn": 2400,
        "n_keep": 200,
        "n_rot": 0,
    }
    inversion = run(
        inversion_kwargs=inversion_kwargs,
        out_dir="/test_inversion_results",
    )


def test_run():
    """
    test with a simple, known model.
    """
    # *** uhhh check units of layer thicknesses
    # *** make sure true model is also fixed
    # *** true model fixed, starting model is with noise.... for simple test.
    true_model_kwargs = {
        "n_layers": 2,
        "layer_bounds": [20e-3, 50e-3],  # km
    }
    inversion_kwargs = {
        "n_layers": 2,
        "n_chains": 1,
    }
    max_perturbations=1,

    param_bounds, true_model = setup_scene(**true_model_kwargs)

    vel_s = [3.50, 4.00] # km/s
    thickness = [27e-3, 40e-3] # in km..
    #true_model.vel_s = vel_s
    true_model.model_params[true_model.n_layers : 2 * true_model.n_layers] = vel_s
    true_model.model_params[: true_model.n_layers] = thickness

    inversion = Inversion(
        true_model.n_data,
        param_bounds,
        true_model.freqs,
        true_model.phase_vel_obs,
        **inversion_kwargs,
    )

    chain = inversion.chains[0]

    # no proper way to set density to a fixed value
    # chain.model_params[-1] = sigma_data
    # chain.model_params[chain.n_layers : 2 * chain.n_layers] = vel_s + 0.10 * np.randn(len(vel_s))
    # chain.model_params[: self.n_layers] = thickness + 5e-3 * np.randn(len(thickness))

    # make a temporary path to save to
    asyncio.get_event_loop().run_until_complete(
        inversion.random_walk(max_perturbations, hist_conv=0.05, out_dir="./tests/out/")
    )

# test convergence

def test_generating_model_params():
    # right now they seem bad.
    # check generating models, what parameters are valid, in bounds
    inversion = run(
        true_model_kwargs={},
        inversion_kwargs={},
        out_dir="/test_inversion_results",
    )
    params = inversion.chains[0].generate_model_params()


def test_lin_rot():
    pass


def test_get_betas():
    # change number of chains and beta_spacing
    # check the fraction of chains that have beta=1
    # later would test that beta updates with acceptance rate

    inversion = setup_inversion_obj()

    # 1/4 to 1/2 of the chains should be beta=1.
    for n_chains in np.range(2, 10):
        inversion.n_chains = n_chains
        betas = inversion.get_betas(beta_spacing_factor=1.15)

        n_betas = np.sum(betas == 1)
        assert n_betas >= 0.25
        assert n_betas <= 0.50 




def test_parallel_computing():
    pass


def test_rotation():
    # save values of running to the end of burn-in, use those to test rotation
    pass


#create_test_file()
test_run()