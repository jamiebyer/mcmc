import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import re
import xarray as xr
from matplotlib.colors import LogNorm
from disba import PhaseDispersion
from inversion.model import Model


def plot_observed_data():
    """
    read dispersion curve
    """

    plt.subplot(2, 1, 1)
    max_file = "./data/WH01/WH01_fine.max"
    txt_path = "./data/WH01/WH01_curve_fine.txt"
    txt_path_2 = "./data/WH01/WH01_curve_fine_2.txt"

    # Open the file in read mode
    with open(max_file, "r") as file:
        # Read the first line
        line = file.readline()
        ind = 0
        while line:
            if "# BEGIN DATA" in line:
                ind += 3
                break
            line = file.readline()  # Read the next line
            ind += 1

    names = [
        "abs_time",
        "frequency",
        "polarization",
        "slowness",
        "azimuth",
        "ellipticity",
        "noise",
        "power",
        "valid",
    ]

    df_max = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)

    freqs = df_max["frequency"]
    vels = 1 / df_max["slowness"]
    # az = df_max["azimuth"]
    # power = df_max["power"]
    #
    plt.hist2d(freqs, vels, bins=200, cmap="coolwarm", norm=LogNorm())
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("freqs")
    plt.ylabel("velocity")

    plt.ylim([0, 1500])

    plt.colorbar()
    plt.grid()

    names = ["frequency", "slowness", "error", "n_binned", "valid"]
    df_txt = pd.read_csv(txt_path, skiprows=5, sep="\s+", names=names)
    df_txt_2 = pd.read_csv(txt_path_2, skiprows=5, sep="\s+", names=names)

    # plt.scatter(df_txt["frequency"], 1 / df_txt["slowness"], c="black", s=5)
    plt.errorbar(
        df_txt["frequency"],
        1 / df_txt["slowness"],
        # 1 / (df_txt["error"] * df_txt["slowness"]),
        1 / df_txt["error"],
        c="black",
    )

    # fit weighted least squares regression model
    # fit_wls = sm.WLS(y, X, weights=wt).fit()

    plt.subplot(2, 1, 2)
    max_file = "./data/WH02/WH02_fine.max"
    txt_path = "./data/WH02/WH02_curve_fine.txt"

    # Open the file in read mode
    with open(max_file, "r") as file:
        # Read the first line
        line = file.readline()
        ind = 0
        while line:
            if "# BEGIN DATA" in line:
                ind += 3
                break
            line = file.readline()  # Read the next line
            ind += 1

    names = [
        "abs_time",
        "frequency",
        "polarization",
        "slowness",
        "azimuth",
        "ellipticity",
        "noise",
        "power",
        "valid",
    ]

    df_max = pd.read_csv(max_file, skiprows=ind, sep="\s+", names=names)

    freqs = df_max["frequency"]
    vels = 1 / df_max["slowness"]
    # az = df_max["azimuth"]
    # power = df_max["power"]
    #
    plt.hist2d(freqs, vels, bins=200, cmap="coolwarm", norm=LogNorm())
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("freqs")
    plt.ylabel("velocity")

    plt.ylim([0, 1500])

    plt.colorbar()
    plt.grid()

    names = ["frequency", "slowness", "error", "n_binned", "valid"]
    df_txt = pd.read_csv(txt_path, sep="\s+", names=names)

    # plt.scatter(df_txt["frequency"], 1 / df_txt["slowness"], c="black", s=5)
    plt.errorbar(
        df_txt["frequency"],
        1 / df_txt["slowness"],
        # 1 / (df_txt["error"] * df_txt["slowness"]),
        1 / df_txt["error"],
        c="black",
    )

    plt.show()


def plot_optimized_model():
    df = pd.read_csv("./results/inversion/optimize_model.csv", index_col=0)

    plt.plot(df["logL"][500:])
    plt.ylabel("logL")
    plt.title("optimize model")
    plt.show()

    thickness = []
    for p in df["thickness"]:
        par = p.removeprefix("[").removesuffix("]")
        par = par.replace("'", "").replace("  ", " ")
        # par = par.split(" ")
        par = par.split(" ")
        par = [i for i in par if i]

        thickness.append(np.array(par, dtype=float))

    vel_s = []
    for p in df["vel_s"]:
        par = p.removeprefix("[").removesuffix("]")
        par = par.replace("'", "").replace("  ", " ")
        # par = par.split(" ")
        par = par.split(" ")
        par = [i for i in par if i]

        vel_s.append(np.array(par, dtype=float))

    vel_p = []
    for p in df["vel_p"]:
        par = p.removeprefix("[").removesuffix("]")
        par = par.replace("'", "").replace("  ", " ")
        # par = par.split(" ")
        par = par.split(" ")
        par = [i for i in par if i]

        vel_p.append(np.array(par, dtype=float))

    density = []
    for p in df["density"]:
        par = p.removeprefix("[").removesuffix("]")
        par = par.replace("'", "").replace("  ", " ")
        # par = par.split(" ")
        par = par.split(" ")
        par = [i for i in par if i]

        density.append(np.array(par, dtype=float))

    m = [0.03] + [0.4, 1.5] + [1.6, 2.5] + [2.0, 2.5]

    plt.subplot(2, 4, 1)
    plt.plot(np.array(thickness)[500:, 0])
    plt.axhline(m[0], c="red")
    plt.ylabel("thickness 1")

    plt.subplot(2, 4, 3)
    plt.plot(np.array(vel_s)[500:, 0])
    plt.axhline(m[1], c="red")
    plt.ylabel("vel s 1")

    plt.subplot(2, 4, 4)
    plt.plot(np.array(vel_s)[500:, 1])
    plt.axhline(m[2], c="red")
    plt.ylabel("vel s 2")

    plt.subplot(2, 4, 5)
    plt.plot(np.array(vel_p)[500:, 0])
    plt.axhline(m[3], c="red")
    plt.ylabel("vel p 1")

    plt.subplot(2, 4, 6)
    plt.plot(np.array(vel_p)[500:, 1])
    plt.axhline(m[4], c="red")
    plt.ylabel("vel p 2")

    plt.subplot(2, 4, 7)
    plt.plot(np.array(density)[500:, 0])
    plt.axhline(m[5], c="red")
    plt.ylabel("density 1")

    plt.subplot(2, 4, 8)
    plt.plot(np.array(density)[500:, 1])
    plt.axhline(m[6], c="red")
    plt.ylabel("density 2")

    plt.suptitle("optimize model")
    plt.show()


def plot_inversion_results_param_time(in_path, skip_inds=0):
    ds = xr.open_dataset(in_path)

    bounds = {
        "thickness": [0.001, 0.1],  # km
        "vel_s": [0.1, 1.8],  # km/s
    }

    m = [0.03] + [0.4, 1.5] + [1.6, 2.5] + [2.0, 2.5]

    model_params = ds["model_params"].values
    step = ds["step"]
    thickness = model_params[ds["thickness_inds"], :]
    vel_s = model_params[ds["vel_s_inds"], :]

    # plt.subplot(4, 2, 1)
    plt.subplot(2, 2, 1)
    plt.scatter(
        np.arange(skip_inds, thickness.shape[1]),
        thickness[0, skip_inds:],
        s=2,
    )
    plt.axhline(m[0], c="red")
    plt.axhline(bounds["thickness"][0], c="black")
    plt.axhline(bounds["thickness"][1], c="black")
    plt.ylabel("thickness 1 (km)")
    plt.xlabel("step")

    # plt.subplot(4, 2, 2)
    plt.subplot(2, 2, 2)
    plt.scatter(np.arange(skip_inds, thickness.shape[1]), vel_s[0, skip_inds:], s=2)
    plt.axhline(m[1], c="red")
    plt.axhline(bounds["vel_s"][0], c="black")
    plt.axhline(bounds["vel_s"][1], c="black")
    plt.ylabel("vel_s 1 (km/s)")
    plt.xlabel("step")
    # plt.subplot(4, 2, 3)
    plt.subplot(2, 2, 3)
    plt.scatter(np.arange(skip_inds, thickness.shape[1]), vel_s[1, skip_inds:], s=2)
    plt.axhline(m[2], c="red")
    plt.axhline(bounds["vel_s"][0], c="black")
    plt.axhline(bounds["vel_s"][1], c="black")
    plt.ylabel("vel_s 2 (km/s)")
    plt.xlabel("step")

    plt.subplot(2, 2, 4)
    # plt.plot(np.arange(skip_inds, ds["thickness"].shape[0]), ds["acc_rate"][skip_inds:])
    # plt.plot(
    #    np.arange(skip_inds, ds["thickness"].shape[0]), ds["err_ratio"][skip_inds:]
    # )
    plt.plot(np.arange(skip_inds, thickness.shape[1]), ds["logL"][skip_inds:])
    # plt.ylabel("")
    # plt.legend(["acc_rate", "err_ratio", "logL"])

    plt.tight_layout()
    plt.suptitle("MCMC model params")

    plt.show()


def plot_inversion_results_param_prob(in_path, skip_inds=0):
    """
    for the histogram, show predicted model visually.
    """
    ds = xr.open_dataset(in_path)

    m = [0.03] + [0.4, 1.5] + [1.6, 2.5] + [2.0, 2.5]

    bounds = {
        "thickness": [0.001, 0.1],  # km
        "vel_s": [0.1, 1.8],  # km/s
    }

    model_params = ds["model_params"].values
    thickness = model_params[ds["thickness_inds"], :]
    vel_s = model_params[ds["vel_s_inds"], :]

    # plt.subplot(4, 2, 1)
    plt.subplot(3, 1, 1)
    plt.hist(thickness[skip_inds:, 0], bins=40, density=True)
    plt.axvline(bounds["thickness"][0], c="black")
    plt.axvline(bounds["thickness"][1], c="black")
    plt.axvline(m[0], c="red")
    plt.xlabel("thickness 1 (km)")

    # plt.subplot(4, 2, 2)
    plt.subplot(3, 1, 2)
    plt.hist(vel_s[0, skip_inds:], bins=40, density=True)
    plt.axvline(bounds["vel_s"][0], c="black")
    plt.axvline(bounds["vel_s"][1], c="black")
    plt.axvline(m[1], c="red")
    plt.xlabel("vel_s 1 (km/s)")

    # plt.subplot(4, 2, 3)
    plt.subplot(3, 1, 3)
    plt.hist(vel_s[1, skip_inds:], bins=40, density=True)
    plt.axvline(bounds["vel_s"][0], c="black")
    plt.axvline(bounds["vel_s"][1], c="black")
    plt.axvline(m[2], c="red")
    plt.xlabel("vel_s 2 (km/s)")

    plt.tight_layout()
    plt.suptitle("MCMC model params")

    plt.show()


def plot_pred_vs_obs(in_path):
    """
    save data separately? in a separate .nc with true data if applicable.
    then the predicted model is read in from the inversion
    should the predicted model be saved directly, or computed from final model.
    """
    ds = xr.open_dataset(in_path)

    # determine the most probable most and use it to run the forward model

    plt.plot(ds["data_obs"])

    # plt.plot(ds.isel(step=slice(-20, -1))["data_pred"].T)

    pd = PhaseDispersion(*velocity_model)
    pd_rayleigh = pd(periods, mode=0, wave="rayleigh")

    plt.legend(["data_obs", "data_true", "data_pred"])
    plt.show()
