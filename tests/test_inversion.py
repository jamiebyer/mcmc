import numpy as np

from plotting.plot_dispersion_curve import *
import os
from inversion.data import SyntheticData
from inversion.model import Model
from inversion.inversion import Inversion
from inversion.model_params import DispersionCurveParams


np.random.seed(0)


def setup_test_data(model_params, noise, depth, vel_s):
    n_data = 50
    periods = np.flip(1 / np.logspace(0, 1.1, n_data))

    # run synthetic data that uses inversion calculations for vel_p and density
    # and optionally, setting vel_p and density exactly.

    data = SyntheticData(
        periods,
        noise,
        model_params,
        depth=depth,
        vel_s=vel_s,
    )

    return data


def setup_test_model(n_layers):
    # set up example data
    proposal_width = {
        "depth": [0.05],
        "vel_s": [0.05],
    }  # fractional step size (multiplied by param bounds width)

    # set up data and inversion params
    bounds = {
        "depth": np.array([0.001, 0.3]),  # km
        # "vel_s": [0.1, 1.8],  # km/s
        "vel_s": np.array([[0.100, 0.500], [0.300, 1.000], [0.750, 2.000]]),  # km/s
    }
    model_params_kwargs = {
        "n_layers": n_layers,
        "vpvs_ratio": 1.75,
        "param_bounds": bounds,
        "proposal_width": proposal_width,
    }
    # model params
    model_params = DispersionCurveParams(**model_params_kwargs)

    return model_params


def basic_inversion(n_layers, noise, sample_prior, set_starting_model, out_filename=""):
    """
    real noise added to synthetic data (percentage)
    assumed noise used in likelihood calculation (percentage)
    """
    sigma_data = noise

    if n_layers == 1:
        # one layer
        depth = [0.02]
        vel_s = [0.4, 1.0]
    elif n_layers == 2:
        # two layers
        depth = [0.02, 0.04]
        vel_s = [0.2, 0.6, 1.0]

    model_params = setup_test_model(n_layers)
    data = setup_test_data(model_params, noise, depth, vel_s)

    inversion_init_kwargs = {
        # "n_burn": 0,
        "n_burn": 1000,
        "n_chunk": 500,
        "n_mcmc": 5000,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
        "out_filename": out_filename,
        "set_starting_model": set_starting_model,
    }

    model_kwargs = {"sigma_data": sigma_data * data.data_obs}

    # run inversion
    inversion = Inversion(
        data,
        model_params,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    return inversion, model_params


# TEST INITIALIZATIONS


def test_data_setup():
    """
    # test that variables are initialized properly
    # compare generated data and true data
    """
    n_layers = 1
    noise = 0.05

    if n_layers == 1:
        # one layer
        depth = [0.02]
        vel_s = [0.4, 1.0]
    elif n_layers == 2:
        # two layers
        depth = [0.02, 0.04]
        vel_s = [0.2, 0.6, 1.0]

    model_params = setup_test_model(n_layers)
    data = setup_test_data(model_params, noise, depth, vel_s)

    freqs = 1 / data.periods
    plt.subplot(2, 1, 1)
    plt.plot(freqs, data.data_true)
    plt.scatter(freqs, data.data_obs)

    plt.plot(freqs, data.data_true + noise * data.data_true, c="black")
    plt.plot(freqs, data.data_true - noise * data.data_true, c="black")

    plt.subplot(2, 1, 2)
    plt.scatter(freqs, np.abs(data.data_true - data.data_obs))
    plt.plot(freqs, noise * data.data_true)

    # plt.title(np.std(data.data_obs))
    plt.show()


# TESTING BAYESIAN INVERSION


def test_perturb_params():
    # check that one parameter is modified at a time
    # check the size of the perturbations....
    # test that each of the params are being modified roughly the same amount.

    inversion, model_params = basic_inversion()

    inversion_run_kwargs = {
        "proposal_distribution": "cauchy",
        "rotate_params": True,
    }
    model = inversion.chains[0]
    # set model params

    model.perturb_params()


def test_run_inversions():
    """
    - Run with sampling prior. Run with setting the starting model, run without.
        - Run with 1 layer, 2 layers.
        - Run with low noise, medium noise, high noise.
    """
    sample_prior = False
    set_starting_model = False
    rotate = False
    n_layers = 2
    noise = 0.02  # 0.02 # 0.05 # 0.1

    inversion, model_params = basic_inversion(
        n_layers=n_layers,
        noise=noise,
        sample_prior=sample_prior,
        set_starting_model=set_starting_model,
        # out_filename=out_filename,
    )
    inversion.random_walk(
        model_params,
        proposal_distribution="cauchy",
        rotate_params=rotate,
    )


def test_linear_rotations():
    """
    need to test linear and PC rotations
    look at the normalized covariance matrix.
    """
    inversion_run_kwargs = {
        "proposal_distribution": "cauchy",
        "rotate_params": True,
    }
    inversion, model_params = basic_inversion()

    model = inversion.chains[0]
    # set model params

    # normalize (by parameter bounds...)
    model_params_norm = (
        model.model_params.model_params - model.param_bounds[:, 0]
    ) / model.param_bounds[:, 2]

    # linear rotation
    model.rotation_matrix, model.posterior_width = model.linear_rotation(
        model_params_norm, inversion.data
    )
    # rotate params
    test_model_params = model.rotation_matrix.T @ model_params_norm
    # *** check the result ***

    # perturb params
    ind = np.random.randint(model.model_params.n_model_params)
    # cauchy distribution
    test_model_params[ind] += model.posterior_width[ind] * np.tan(
        np.pi * (np.random.uniform() - 0.5)
    )

    # rotate back
    test_model_params = model.rotation_matrix @ test_model_params

    # reverse normalization (to check forward model)
    # could be before or after checking bounds...
    test_model_params = (
        test_model_params * model.param_bounds[:, 2]
    ) + model.param_bounds[:, 0]

    # *** check that the results make sense ***


def test_PC_rotations():
    """
    need to test linear and PC rotations
    look at the normalized covariance matrix.
    """
    # inversion = basic_inversion()
    inversion, model_params, inversion_run_kwargs = basic_inversion()
    model = inversion.chains[0]

    # set model params
    # rotate
    # check that results make sense...
    # rotate back
    # check that results make sense...
    # normalize (by parameter bounds...)
    model_params_norm = (
        model.model_params.model_params - model.param_bounds[:, 0]
    ) / model.param_bounds[:, 2]

    # get rotation matrix and step size from correlation matrix
    model.rotation_matrix, model.posterior_width = model.update_covariance_matrix(
        n_step, model_params_norm
    )

    # *** pick a random parameter to perturb
    ind = np.random.randint(model.model_params.n_model_params)

    test_model_params = model.rotation_matrix.T @ model_params_norm

    # cauchy distribution
    test_model_params[ind] += model.posterior_width[ind] * np.tan(
        np.pi * (np.random.uniform() - 0.5)
    )

    # rotate back
    test_model_params = model.rotation_matrix @ test_model_params

    # reverse normalization (to check forward model)
    # could be before or after checking bounds...
    test_model_params = (
        test_model_params * model.param_bounds[:, 2]
    ) + model.param_bounds[:, 0]

    # check bounds
    # valid_params = self.validate_bounds(test_model_params)


def test_layer_swap():
    # propose parameters where the lower layer crosses the upper layer
    # create inversion object
    # use sort layers / perturb params, but give the new params.
    # swaps and new swap isn't in bounds?
    # swap and is accepted, swap and not accepted...

    sample_prior = False
    set_starting_model = True
    rotate = False
    n_layers = 2
    noise = 0.02  # 0.02 # 0.05 # 0.1

    inversion, model_params = basic_inversion(
        n_layers=n_layers,
        noise=noise,
        sample_prior=sample_prior,
        set_starting_model=set_starting_model,
        # out_filename=out_filename,
    )

    # propose perturbation
    # swap depths
    model = inversion.chains[0]
    test_model_params = model.model_params.model_params.copy()
    p1 = test_model_params.copy()
    """
    depths = test_model_params[model.model_params.depth_inds]
    depth_1 = depths[0]
    depth_2 = depths[1]
    depths[0:2] = [depth_2, depth_1]
    test_model_params[model.model_params.depth_inds] = depths
    """
    test_model_params[0] = 0.04
    test_model_params[1] = 0.02

    p2 = test_model_params.copy()
    # sort layers
    test_model_params = model.model_params.sort_layers(test_model_params)
    p3 = test_model_params.copy()

    print(p1)
    print(p2)
    print(p3)

    assert p1 == p3


def test_acceptance_criteria():
    # try proposing a model out of bounds
    # rejection records the same model twice
    # test that all variables have an acceptance rate of ~0.3
    # compare the start of the run and the end of the run
    pass


def test_proposal_distribution():
    # run an inversion and check that the results are roughly uniform
    # and in bounds.
    pass


def test_likelihood():
    # test the computation...
    # make sure the likelihood is negative
    # make sure it's increasing with time
    # validate the specific computation somehow...

    # for starting at the true model,
    # We expect that the sampler will find some logL values
    # that are better than those of the true model. However, most will be lower in logL.

    # for low noise model
    pass


def test_writing_samples():
    # check that it is not overwriting file
    # check that after writing, it reads in what you expect...
    pass


def test_most_probable_model():
    # test that the most probable model is computed properly
    pass


# TESTING OPTIMIZATION INVERSION


def test_constant_temp_inversion(setup_data):
    data = setup_data
    bounds = {
        "thickness": [0.01, 1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 5],  # km/s
        "density": [0.5, 5],  # g/cm^3
    }
    model_kwargs = {
        "n_layers": 2,
        "poisson_ratio": 0.265,
        "sigma_model": {"thickness": 0.1, "vel_s": 0.01},  # km  # km/s
        "beta": 1,
        "n_bins": 200,
    }
    model = Model(**model_kwargs)

    model.thickness = np.random.uniform(
        bounds["thickness"][0], bounds["thickness"][1], 1
    )

    model.vel_s = np.random.uniform(bounds["vel_s"][0], bounds["vel_s"][1], 2)
    valid_params = False
    while not valid_params:
        velocity_model, valid_params = model.get_velocity_model(
            bounds, model.thickness, model.vel_s
        )

    # set initial likelihood
    model.logL, model.data_pred = model.get_likelihood(velocity_model, data)
    # run inversion optimization
    model.get_optimization_model(bounds, data)

    # check acceptance rate over time?
    # check that it converges


def test_optimization_inversion(setup_data):
    data = setup_data
    bounds = {
        "thickness": [0.01, 1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 5],  # km/s
        "density": [0.5, 5],  # g/cm^3
    }
    model_kwargs = {
        "n_layers": 2,
        "poisson_ratio": 0.265,
        "sigma_model": {"thickness": 0.1, "vel_s": 0.01},  # km  # km/s
        "beta": 1,
        "n_bins": 200,
    }
    model = Model(**model_kwargs)

    model.thickness = np.random.uniform(
        bounds["thickness"][0], bounds["thickness"][1], 1
    )

    model.vel_s = np.random.uniform(bounds["vel_s"][0], bounds["vel_s"][1], 2)
    valid_params = False
    while not valid_params:
        velocity_model, valid_params = model.get_velocity_model(
            bounds, model.thickness, model.vel_s
        )

    # set initial likelihood
    model.logL, model.data_pred = model.get_likelihood(velocity_model, data)
    # run inversion optimization
    model.get_optimization_model(bounds, data)

    # check acceptance rate over time?
    # check that it converges
