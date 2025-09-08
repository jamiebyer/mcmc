import cProfile
from pstats import Stats, SortKey
import os
import numpy as np

from inversion.data import SyntheticData
from inversion.model_params import DispersionCurveParams
from inversion.inversion import Inversion

from plotting.plot_dispersion_curve import *

import xarray as xr


def plot_inversion(file_name):

    input_path = "./results/inversion/input-" + file_name + ".nc"
    results_path = "./results/inversion/results-" + file_name + ".nc"

    input_ds = xr.open_dataset(input_path)
    results_ds = xr.open_dataset(results_path)

    plot_results(input_ds, results_ds, out_filename=file_name)

    # plot_covariance_matrix(input_ds, results_ds)
    # model_params_timeseries(input_ds, results_ds, save=True, out_filename=file_name)
    # model_params_autocorrelation(
    #     input_ds, results_ds, save=False, out_filename=file_name
    # )
    # model_params_histogram(input_ds, results_ds, save=True, out_filename=file_name)
    # resulting_model_histogram(input_ds, results_ds, save=True, out_filename=file_name)
    # plot_data_pred_histogram(input_ds, results_ds, save=True, out_filename=file_name)
    # plot_likelihood(input_ds, results_ds, save=True, out_filename=file_name)
    # save_inversion_info(input_ds, results_ds)


if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """

    # run_inversion()

    file_name = "1757354761"
    plot_inversion(file_name)
