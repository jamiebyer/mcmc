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
                            depth = [t1, t1 + t2]
                            vel_s = [v1, v2, v3]
                            d_list.append(depth)
                            v_list.append(vel_s)
    return d_list[ind], v_list[ind]


def basic_inversion(
    ind,
    n_layers,
    noise_dist,
    noise_params,
    inv_noise_dist,
    inv_noise_params,
    set_starting_model,
    set_random_seed,
    save_essentials,
    out_dir,
    out_filename,
):
    """
    real noise added to synthetic data (percentage)
    assumed noise used in likelihood calculation (percentage)
    """

    inversion_init_kwargs = {
        "n_burn": 100,
        "n_chunk": 500,
        "n_mcmc": 1000,
        "n_cov_chunk": 500,
        "n_thin": 1000,
        "n_chains": 1,
        "beta_spacing_factor": 1.15,
        "set_starting_model": set_starting_model,
        "set_random_seed": set_random_seed,
        "save_essentials": save_essentials,
        "out_dir": out_dir,
        "out_filename": out_filename,
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

    (
        freqs_2d,
        noise_2d,
        AL_q_lower,
        AL_q_higher,
        norm_q_lower,
        norm_q_higher,
        stds,
    ) = data.generate_noise_dist()
    # plot synthetic data
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
    # """
    # raise ValueError

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
            inv_noise_params["lambd"] = 39.13138936314191
        elif noise_params["std"] == 0.075:
            inv_noise_params["lambd"] = 12.273994073235064
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


def run_inversion(
    n_layers=2,
    # noise_dist="normal",
    noise_dist="asym-laplace",
    # inv_noise_dist="normal",
    inv_noise_dist="asym-laplace",
    lambd_scaling=False,
    kappa_scaling=True,
    sample_prior=False,
    set_starting_model=True,
    rotate=False,
    std=None,  # 0.075  # km/s
    std_percent=None,  # 0.10
    lambd_scale=None,
    lambd=None,
    lambd_scale=None,
    kappa=None,
    kappa_scale=None,
    set_random_seed=True,
    save_essentials=False,
    out_dir="./results/inversion/",
    out_filename="",
    ind=None,
):
    """
    - Run with sampling prior. Run with setting the starting model, run without.
    - Run with 1 layer, 2 layers.
    - Run with low noise, medium noise, high noise.
    """
    noise_params = {"lambd_scaling": lambd_scaling, "kappa_scaling": kappa_scaling}
    if noise_dist == "normal":
        # for normal errors with frequency dependence,
        # use the percent of the data as the standard deviation
        # noise_params["std_percent"] = std_percent
        
        # for IID errors, the value for normal standard deviation
        noise_params["std"] = std

    elif noise_dist == "asym-laplace":
        # lambd_scale = 1.0  # 0.055 # 0.130 # 0.200 # km/s
        # lambd, kappa = 14.03508769968587, 1.0 # normal std = 0.025
        
        # lambd, kappa = 39.13138936314191, 1.0 # std=0.025
        # lambd, kappa = 12.273994073235064, 1.0 # std=0.075

        # lambd, kappa = 1.0, 1.0

        
        # kappa_scaling
        lambd, kappa = 63.21939218002945, 1.0

        # both_scaling
        # lambd, kappa = 1.0, 1.0
        

        noise_params["lambd"] = lambd
        noise_params["kappa"] = kappa

        if lambd_scaling or kappa_scaling:
            # df = pd.read_csv("./data/remove_artifact_True/both_scaling/WH01-asym-laplace-params.csv")
            df = pd.read_csv("./data/remove_artifact_True/kappa_scaling/WH01-asym-laplace-params.csv")
            noise_params["scale_freqs"] = df["freqs"].values
            if lambd_scaling:  
                noise_params["lambd_scale"] = np.flip(df["scaled_lambd"].values.copy())
                # noise_params["lambd_scale"][noise_params["lambd_scale"] > 60] = 60 # WH01
            if kappa_scaling:
                noise_params["kappa_scale"] = np.flip(df["scaled_kappa"].values.copy())
        else:
            # IID lambda scale
            noise_params["lambd_scale"] = lambd_scale

    # inv_noise_params = {"frequency_scaling": noise_params["frequency_scaling"]}
    inv_noise_params = {
        "lambd_scaling": noise_params["lambd_scaling"], 
        "kappa_scaling": noise_params["kappa_scaling"]
    }
    # inv_noise_params = noise_params.copy()

    # currently set up to use same noise params for real noise and for model noise
    inversion, model_params = basic_inversion(
        ind,
        n_layers=n_layers,
        noise_dist=noise_dist,
        noise_params=noise_params,
        inv_noise_dist=inv_noise_dist,
        inv_noise_params=inv_noise_params,
        set_starting_model=set_starting_model,
        set_random_seed=set_random_seed,
        save_essentials=save_essentials,
        out_dir=out_dir,
        out_filename=out_filename,
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


def run_inversion_ensemble(job_id, task_id, n_runs):
    # print(job_id, task_id)
    inv_noise_dist = "normal" # "asym-laplace"
    # kappa = 1.0 # 0.75
    kappa = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    kappa_list = []
    for k in kappa:
        kappa_list += n_runs * [k]

    # lambd = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    lambd = 1.0

    # lambd_list = []
    # for l in lambd:
    #     lambd_list += n_runs * [l]

    # out_dir = "/scratch/" + str(job_id) + "/"
    out_dir = "./results/inversion/"#  + str(job_id) + "/"
    # os.mkdir(out_dir)

    # out_filename = str(kappa) + "/" + inv_noise_dist + "/" + str(lambd_list[task_id]) + "-" + str(task_id)
    out_filename = "l-" + str(lambd) + "/" + inv_noise_dist + "/" + str(kappa_list[task_id]) + "-" + str(task_id)
    # out_filename = str(kappa) + "/" + inv_noise_dist + "/" + str(lambd) + "-" + str(task_id)
    # out_filename = str(kappa) + "-" + inv_noise_dist + "-" + str(lambd) + "-" + str(task_id)
    # print(out_filename)

    run_inversion(
        n_layers=2,
        # noise_dist="normal",
        noise_dist="asym-laplace",
        # inv_noise_dist="normal",
        # inv_noise_dist="asym-laplace",
        inv_noise_dist=inv_noise_dist,
        frequency_scaling=True,
        sample_prior=False,
        set_starting_model=True,
        rotate=False,
        std=None,  # 0.075  # km/s
        std_percent=None,  # 0.10
        lambd_scale=None,
        # lambd=lambd_list[task_id],
        lambd=lambd,
        # kappa=kappa,
        kappa=kappa_list[task_id],
        set_random_seed=False,
        save_essentials=True,
        out_dir=out_dir,
        out_filename=out_filename,
        ind=None,
    )


def assemble_ensemble_files(job_id):
    inv_noise_dist = "asym-laplace" # "normal"
    kappa = 0.75 # 1.0
    lambd = 0.5

    out_dir = "/scratch/" + str(job_id) + "/"
    out_filename = str(kappa) + "-" + inv_noise_dist + "-" + str(lambd)

    input_paths, results_paths = [], []
    input_files, results_files = [], []
    print(os.listdir(out_dir), "\n")
    print(out_filename, "\n")
    for f in os.listdir(out_dir):
        if "input-" in f and out_filename in f:
            input_paths.append(out_dir + f)
            input_files.append(xr.open_dataset(out_dir + f))
        elif "results-" in f and out_filename in f:
            results_paths.append(out_dir + f)
            results_files.append(xr.open_dataset(out_dir + f))

    # ds = xr.concat(input_files, dim="ind", combine_attrs="identical")
    # print(ds)
    # print("\n", ds.attrs.keys())
    # print("\n", ds.dims.keys())
    # ds.to_netcdf("results/inversion/" + str(kappa) + "/" + inv_noise_dist + "/" + "input-" + str(lambd))
    # ds.to_zarr("results/inversion/" + str(kappa) + "/" + inv_noise_dist + "/" + "input-" + str(lambd))

    print(len(results_files))
    # ds = xr.concat(results_files, dim="ind")
    # ds.to_netcdf("results/inversion/" + str(kappa) + "/" + inv_noise_dist + "/" + "results-" + str(lambd))
    # ds.to_zarr("results/inversion/" + str(kappa) + "/" + inv_noise_dist + "/" + "results-" + str(lambd))

    # loop over saved paths and delete them
    # [os.remove(f) for f in input_paths]
    # [os.remove(f) for f in results_paths]


def plot_ensemble_inversions():
    inv_noise_dist = "asym-laplace"
    # inv_noise_dist = "normal"
    kappa = 1.0 # 0.75
    # kappa = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # lambd = 1.0
    lambd = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    # file_list = [inv_noise_dist + "-" + str(kappa) + "-" + str(l) for l in lambd]

    dir_path = "./results/inversion/" + str(kappa) + "/" + inv_noise_dist
    # dir_path = "./results/inversion/l-" + str(lambd) + "/" + inv_noise_dist
    input_list, results_list = [], []
    for f in os.listdir(dir_path):
        if "input-" in f:
            input_list.append(xr.open_dataset(dir_path + "/" + f))
            # input_list.append(xr.open_zarr(dir_path + "/" + f))
        elif "results-" in f:
            results_list.append(xr.open_dataset(dir_path + "/" + f))
            # results_list.append(xr.open_zarr(dir_path + "/" + f))

    # plot generated observed data
    plot_data_obs_ensemble(
        input_list, 
        # results_ds_list,
        # attrs=kappa,
        attrs=lambd,
        save=True, 
        out_filename=inv_noise_dist + "-k-" + str(kappa)
        # out_filename=inv_noise_dist + "-l-" + str(lambd)
    )

    # plot model params
    """
    model_params_histogram_ensemble(
        # input_ds_list=[xr.open_dataset("./results/inversion/input-" + f + ".nc") for f in file_list],
        # results_ds_list=[xr.open_dataset("./results/inversion/results-" + f + ".nc") for f in file_list],
        input_ds_list=input_list,
        results_ds_list=results_list,
        save=True,
        plot_true_model=True,
        attrs=lambd,
    )
    """

if __name__ == "__main__":
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """
    # job_id = int(sys.argv[1])
    # task_id = int(sys.argv[2])
    # run_inversion_ensemble(job_id, task_id, n_runs=10)

    # assemble_ensemble_files(45017762)
    
    # plot_ensemble_inversions()


    run_inversion()

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
    # std=0.075
    # file_name = "1781113173"

    # model 4
    # 0.055: 0.02456599088986876
    # noise: AL, model: AL
    # file_name = "1781626367"
    # noise: AL, model: normal
    # file_name = "1781626685"

    # 0.200: 0.07472376455521576
    # noise: AL, model: AL
    # file_name = "1781627173"
    # noise: AL, model: normal
    # file_name = "1781627461"

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

    # reverse mismatch
    # noise: normal, inv_noise: AL
    # std = 0.025, lambd = 39.13138936314191
    # file_name = "1782437945"
    # std = 0.075, lambd = 12.273994073235064
    # file_name = "1782438122"

    # normal cases
    # noise: normal, inv_noise: normal
    # std = 0.025
    # file_name = "1782486847"
    # std = 0.075
    # file_name = "1782487008"

    # frequency scaling WH01
    # noise AL, model: AL
    # file_name = "1782935477"
    # noise AL, model: normal
    # file_name = "1782946945"

    # frequency scaling WH04
    # floor: 90
    # noise AL, model: AL
    # file_name = "1783012215"
    # noise AL, model: normal
    # file_name = "1783012681"

    # floor: 75
    # noise AL, model: AL
    # file_name = "1783016338"
    # noise AL, model: normal
    # file_name = "1783016057"

    # kappa scaling
    file_name = "1786303466"


    plot_inversion(file_name)
