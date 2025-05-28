from inversion.forward_model import SyntheticData, FieldData, GeneratedData
from plotting.inversion_plotting import *
from plotting.data_plotting import *
from inversion.inversion import Inversion
from fk_processing.read_mseed import *
import asyncio
import numpy as np
from tests.test_inversion import *


def run_inversion(data_type="field_data"):
    np.random.seed(0)

    bounds = {
        "thickness": [0.001, 0.1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 1.8],  # km/s
        "density": [0.5, 3],  # g/cm^3
    }
    model_kwargs = {
        "n_layers": 2,
        "sigma_model": {"thickness": 0.1, "vel_s": 0.01},  # km  # km/s
        "poisson_ratio": 0.265,
        "param_bounds": bounds,
    }
    inversion_init_kwargs = {
        "n_bins": 200,
        "n_burn": 0,
        "n_keep": 100,
        "n_rot": 0,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
    }
    inversion_run_kwargs = {
        "max_perturbations": 10,
        "hist_conv": 0.05,
    }

    n_data = 50
    periods = np.flip(1 / np.logspace(-2, 2, n_data))
    sigma_data = 0.01
    if data_type == "synthetic_data":
        data_kwargs = {
            "thickness": [0.03],
            "vel_s": [0.4, 1.5],
            "vel_p": [1.6, 2.5],
            "density": [2.0, 2.5],
        }
        data = SyntheticData(periods, sigma_data, **data_kwargs)
    elif data_type == "generated_data":
        data = GeneratedData(periods, sigma_data, bounds, model_kwargs["n_layers"])
    elif data_type == "field_data":
        path = "./results/WH01/WH01_main.max"
        data = FieldData(path)

    # run inversion
    inversion = Inversion(
        data,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    inversion.random_walk(**inversion_run_kwargs)
    # asyncio.get_event_loop().run_until_complete(
    #    inversion.random_walk(**inversion_run_kwargs)
    # )


if __name__ == "__main__":
    # in_path = "./results/inversion/results1745810948.nc"
    in_path = "./results/inversion/results1745812372.nc"

    # run_inversion()

    # plot_dispersion_curve()
    # plot_array_response()

    # plot_optimized_model()
    # plot_array_response()

    # plot_inversion_results_param_prob(in_path)
    # plot_inversion_results_param_time(in_path)
