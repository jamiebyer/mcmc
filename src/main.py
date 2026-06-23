import cProfile
from pstats import Stats, SortKey
import sys
import os
from time import sleep

import numpy as np
import pandas as pd

from inversion.data import SyntheticData
from inversion.model_params import DispersionCurveParams
from inversion.inversion import Inversion

from plotting.plot_dispersion_curve import *

# from plotting.plot_dispersion_curve_plotly import *

import xarray as xr

np.random.seed(0)


def setup_test_data(model_params, noise_dist, noise_params, depth, vel_s):
    if "scale_freqs" in noise_params:
        periods = np.flip(1 / noise_params["scale_freqs"])
    else:
        n_data = 40
        # periods = np.flip(1 / np.logspace(-1, 1.2, n_data))
        periods = np.flip(1 / np.logspace(0.3, 1.3, n_data))

    # run synthetic data that uses inversion calculations for vel_p and density
    # and optionally, setting vel_p and density exactly.

    data = SyntheticData(
        periods,
        noise_dist,
        noise_params,
        model_params,
        depth=depth,
        vel_s=vel_s,
    )

    return data


def setup_test_model(n_layers):
    # set up example data
    proposal_width = {
        "depth": 0.05,
        "vel_s": 0.05,
    }  # fractional step size (multiplied by param bounds width)

    # set up data and inversion params
    if n_layers == 1:
        bounds = {
            "depth": np.array([0.001, 0.15]),  # km
            # "vel_s": [0.1, 1.8],  # km/s
            "vel_s": np.array([[0.100, 0.750], [0.500, 2.000]]),  # km/s
        }
    elif n_layers == 2:
        bounds = {
            "depth": np.array([0.001, 0.10]),  # km
            # model 1
            "vel_s": np.array([[0.100, 0.700], [0.100, 0.700], [1.000, 2.000]]),  # km/s
            # model 2
            # "vel_s": np.array([0.050, 2.000]),  # km/s
        }
    elif n_layers == 3:
        bounds = {
            "depth": np.array([0.001, 0.15]),  # km
            "vel_s": np.array([0.100, 2.000]),  # km/s
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


def generate_model(ind):
    """
    noise: [25, 75]
    thickness 1: [10, 20, 30]
    thickness 2: [10, 20, 30, 40, 50, 60]
    vel s 1: [150, 200, 300, 400, 500] 
    vel s 2: [300, 400, 500, 600, 700, 800]
    vel s 3: [600, 800, 1000, 1200, 1500]
    """
    d_list = []
    v_list = []
    for t1 in [0.010, 0.020, 0.030]:
        for t2 in [0.010, 0.020, 0.030, 0.040, 0.050, 0.060]:
            for v1 in [0.150, 0.200, 0.300, 0.400, 0.500]:
                for v2 in [0.300, 0.400, 0.500, 0.600, 0.700, 0.800]:
                    for v3 in [0.600, 0.800, 1.000, 1.200, 1.500]:
                        if v1 < v2 and v2 < v3:
                            depth = [t1, t1+t2]
                            vel_s = [v1, v2, v3]
                            d_list.append(depth)
                            v_list.append(vel_s)
    print(len(d_list))
    return d_list[ind], v_list[ind]

def basic_inversion(
    ind,
    n_layers,
    noise_dist,
    noise_params,
    inv_noise_dist,
    inv_noise_params,
    set_starting_model,
):
    """
    real noise added to synthetic data (percentage)
    assumed noise used in likelihood calculation (percentage)
    """

    inversion_init_kwargs = {
        "n_burn": 10000,
        "n_chunk": 500,
        "n_mcmc": 50000,
        "n_cov_chunk": 500,
        "n_thin": 10,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
        "set_starting_model": set_starting_model,
    }

    if n_layers == 1:
        # one layer
        depth = [0.05]
        vel_s = [0.4, 1.0]
    elif n_layers == 2:
        # two layers
        # model 1
        if ind == None:
            depth = [0.010, 0.063]
            vel_s = [0.200, 0.400, 1.700]
        else:
            depth, vel_s = generate_model(ind)

    elif n_layers == 3:
        # three layers
        depth = [0.02, 0.04, 0.1]
        vel_s = [0.2, 0.6, 1.0, 1.5]

    model_params = setup_test_model(n_layers)
    data = setup_test_data(model_params, noise_dist, noise_params, depth, vel_s)

    # plot synthetic data
    (
        freqs_2d,
        noise_2d,
        AL_q_lower,
        AL_q_higher,
        norm_q_lower,
        norm_q_higher,
        stds,
    ) = data.generate_noise_dist()
    """
    data.plot_simulated_data_hist2d(
        freqs_2d,
        noise_2d,
        AL_q_lower,
        AL_q_higher,
        norm_q_lower,
        norm_q_higher,
        stds,
    )
    data.plot_simulated_data_frequencies(
        freqs_2d,
        noise_2d,
        AL_q_lower,
        AL_q_higher,
        norm_q_lower,
        norm_q_higher,
        stds,
    )
    """

    # use synthetic noise dist to define normal model noise params
    if noise_dist == "asym-laplace" and inv_noise_dist == "normal":
        (
            freqs_2d,
            noise_2d,
            AL_q_lower,
            AL_q_higher,
            norm_q_lower,
            norm_q_higher,
            stds,
        ) = data.generate_noise_dist()
        inv_noise_params["std"] = stds

        """
        SyntheticData.plot_simulated_data_hist2d(
            data.periods,
            data.data_true,
            data.data_obs,
            freqs_2d,
            noise_2d,
            AL_q_lower,
            AL_q_higher,
            norm_q_lower,
            norm_q_higher,
        )
        """
    elif noise_dist == "normal" and inv_noise_dist == "asym-laplace":
        inv_noise_params["kappa"] = 1.0
        inv_noise_params["lambd_scale"] = 1
        if noise_params["std"] == 0.025:
            inv_noise_params["lambd"] = 14.03508769968587
        elif noise_params["std"] == 0.075:
            inv_noise_params["lambd"] = 4.622714571212673
    elif noise_dist == inv_noise_dist:
        inv_noise_params = noise_params.copy()

    # for frequency dependent noise model, scale using observed data
    # for percent scaling only
    """
    if inv_noise_params["frequency_scaling"]:
        if inv_noise_dist == "normal":
            inv_noise_params["std"] = inv_noise_params["std_percent"] * data.data_obs
        elif inv_noise_dist == "asym-laplace":
            inv_noise_params["lambd_scale"] = (
                inv_noise_params["lambd_scale_percent"] * data.data_obs
            )
    """
    model_kwargs = {"noise_dist": inv_noise_dist, "noise_params": inv_noise_params}

    """
    df = pd.DataFrame(
        {
            "frequency": 1 / data.periods,
            "data_obs": data.data_obs,
            "std": inv_noise_params["std"],
        }
    )
    df.to_csv("./results/data_obs.csv")
    """

    # run inversion
    inversion = Inversion(
        data,
        model_params,
        **model_kwargs,
        **inversion_init_kwargs,
    )

    return inversion, model_params


def get_noise_params():
    pass


def run_inversion(ind=None):
    """
    - Run with sampling prior. Run with setting the starting model, run without.
    - Run with 1 layer, 2 layers.
    - Run with low noise, medium noise, high noise.
    """
    sample_prior = False
    set_starting_model = True
    rotate = False

    n_layers = 2
    # noise_dist = "normal"
    noise_dist = "asym-laplace"
    # inv_noise_dist = "normal"
    inv_noise_dist = "asym-laplace"
    frequency_scaling = False

    noise_params = {"frequency_scaling": frequency_scaling}
    if noise_dist == "normal":
        # std = 0.0001  # km/s
        std = 0.075  # km/s
        # std = 0.07472376455521576
        std_percent = 0.10

        if frequency_scaling:
            # for normal errors with frequency dependence,
            # use the percent of the data as the standard deviation
            noise_params["std_percent"] = std_percent
        else:
            # for IID errors, the value for normal standard deviation
            noise_params["std"] = std

    elif noise_dist == "asym-laplace":
        lambd_scale = 1.0  # 0.055 # 0.130 # 0.200 # km/s
        # lambd_scale_percent = 0.10
        # lambd, kappa = 14.03508769968587, 1.0 # normal std = 0.025
        lambd, kappa = 3.6018791201809166, 0.8638603408785489 # site WH04

        # lambd, kappa = 5.6, 1.50
        # lambd, kappa = 5.6, 0.72
        # lambd, kappa = 5.6, 1.0

        noise_params["lambd"] = lambd
        noise_params["kappa"] = kappa
        if frequency_scaling:
            # scaling by percent
            # noise_params["lambd_scale_percent"] = lambd_scale_percent
            # scaling by field data
            df = pd.read_csv("./data/spread/WH04.csv")
            noise_params["scale_freqs"] = df["freq"].values
            noise_params["lambd_scale"] = 1 / df["spread"].values
        else:
            # IID lambda scale
            noise_params["lambd_scale"] = lambd_scale
            """
            # scaling by field data
            df = pd.read_csv("./data/spread/WH04.csv")
            noise_params["scale_freqs"] = df["freq"].values
            noise_params["lambd_scale"] = df["spread"].values
            """
    inv_noise_params = {"frequency_scaling": noise_params["frequency_scaling"]}

    # currently set up to use same noise params for real noise and for model noise
    inversion, model_params = basic_inversion(
        ind,
        n_layers=n_layers,
        noise_dist=noise_dist,
        noise_params=noise_params,
        inv_noise_dist=inv_noise_dist,
        inv_noise_params=inv_noise_params,
        set_starting_model=set_starting_model,
    )
    inversion.random_walk(
        model_params,
        proposal_distribution="cauchy",
        sample_prior=sample_prior,
        rotate_params=rotate,
    )


def plot_inversion(file_name):
    input_path = "./results/inversion/input-" + file_name + ".nc"
    results_path = "./results/inversion/results-" + file_name + ".nc"

    input_ds = xr.open_dataset(input_path)
    results_ds = xr.open_dataset(results_path)

    plot_results(input_ds, results_ds, out_filename=file_name, plot_true_model=True)


def plot_compare(file_names):
    input_ds_list, results_ds_list = [], []
    for f in file_names:
        input_path = "./results/inversion/input-" + f + ".nc"
        results_path = "./results/inversion/results-" + f + ".nc"

        input_ds = xr.open_dataset(input_path)
        results_ds = xr.open_dataset(results_path)

        input_ds_list.append(input_ds)
        results_ds_list.append(results_ds)

    model_params_histogram_compare(
        input_ds_list,
        results_ds_list,
        save=True,
        plot_true_model=True,
    )


if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """
    #ind = int(sys.argv[1])
    # for ind in range(23, 50):
    # for ind in range(500, 525):
    #     run_inversion(ind)
    #    sleep(1)

    # run_inversion()

    # 1 layer
    # normal IID
    # std = 0.050
    # file_name = "1778191004"

    # AL IID
    # lambd, kappa = 5.6, 1.0, lambd_scale = 0.200
    # file_name = "1778191173"

    # 2 layers
    # model 4, std=0.025
    # file_name = "1781113055"
    # std=0.050
    # file_name = "1781115461"
    # std=0.075
    # file_name = "1781113173"

    # model 4
    # 0.055: 0.02456599088986876
    # noise: AL, model: AL
    # file_name = "1781626367"
    # noise: AL, model: normal
    # file_name = "1781626685"

    # 0.130: 0.04999076164489357
    # noise: AL, model: AL
    # file_name = "1781626752"
    # noise: AL, model: normal
    # file_name = "1781626996"

    # 0.200: 0.07472376455521576
    # noise: AL, model: AL
    # file_name = "1781627173"
    # noise: AL, model: normal
    # file_name = "1781627461"


    # original bandwidth
    # kappa: 2.00
    # 0.055: 0.02456599088986876
    # noise: AL, model: AL
    # file_name = "1781637667"
    # noise: AL, model: normal
    # file_name = "1781637987"

    # kappa: 1.50
    # noise: AL, model: AL
    # file_name = "1781638124"
    # noise: AL, model: normal
    # file_name = "1781638435"

    # shorter bandwidth
    # kappa = 0.50, lambd_scale = 0.055
    # noise: AL, model: AL
    # file_name = "1781732831"
    # noise: AL, model: normal
    # file_name = "1781733045"

    # kappa = 0.50, lambd_scale = 0.200
    # noise: AL, model: AL
    # file_name = "1781733273"
    # noise: AL, model: normal
    # file_name = "1781733486"

    # kappa = 0.25, lambd_scale = 0.055
    # noise: AL, model: AL
    # file_name = "1781735471"
    # noise: AL, model: normal
    # file_name = "1781735660"

    # kappa = 0.25, lambd = 5.6*0.200
    # noise: AL, model: AL
    # file_name = "1781735993"
    # noise: AL, model: normal
    # file_name = "1781736155"

    # saving dist
    # kappa = 0.25, lambd = 5.6*0.055
    # noise: AL, model: AL
    # file_name = "1781814326"
    # noise: AL, model: normal
    # file_name = "1781819964"

    # normal std: 0.025, kappa = 0, lambd = 14.03508769968587
    # noise: normal, model: normal
    # file_name = "1782169807"
    # noise: normal, model: AL
    # file_name = "1782171590"
    # file_name = "1782236521"

    # normal std: 0.075, kappa = 0, lambd = 4.622714571212673
    # noise: normal, model: normal
    # file_name = "1782172312"
    # noise: normal, model: AL
    # file_name = "1782173269"
    # file_name = "1782235298"
    # file_name = "1782236648"

    # frequency scaled
    # file_name = "1782238926"
    file_name = "1782239972"

    plot_inversion(file_name)
