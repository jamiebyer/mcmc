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
from inversion.model_params import DispersionCurveParams
from matplotlib.gridspec import GridSpec


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
    # *** plot acceptance rate and likelihood ***
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

    _, _, prob_params = get_probable_model(in_path)

    # plt.subplot(4, 2, 1)
    plt.subplot(2, 2, 1)
    plt.scatter(
        np.arange(skip_inds, thickness.shape[1]),
        thickness[0, skip_inds:],
        s=2,
    )
    plt.axhline(m[0], c="red")
    plt.axhline(prob_params[0], c="purple")
    plt.axhline(bounds["thickness"][0], c="black")
    plt.axhline(bounds["thickness"][1], c="black")
    plt.ylabel("thickness 1 (km)")
    plt.xlabel("step")

    # plt.subplot(4, 2, 2)
    plt.subplot(2, 2, 2)
    plt.scatter(np.arange(skip_inds, thickness.shape[1]), vel_s[0, skip_inds:], s=2)
    plt.axhline(m[1], c="red")
    plt.axhline(prob_params[1], c="purple")
    plt.axhline(bounds["vel_s"][0], c="black")
    plt.axhline(bounds["vel_s"][1], c="black")
    plt.ylabel("vel_s 1 (km/s)")
    plt.xlabel("step")

    # plt.subplot(4, 2, 3)
    plt.subplot(2, 2, 3)
    plt.scatter(np.arange(skip_inds, thickness.shape[1]), vel_s[1, skip_inds:], s=2)
    plt.axhline(m[2], c="red")
    plt.axhline(prob_params[2], c="purple")
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

    _, _, prob_params = get_probable_model(in_path)

    # plt.subplot(4, 2, 1)
    plt.subplot(3, 1, 1)
    plt.hist(thickness[0, skip_inds:], bins=40, density=True)
    plt.axvline(bounds["thickness"][0], c="black")
    plt.axvline(bounds["thickness"][1], c="black")
    plt.axvline(m[0], c="red")
    plt.axvline(prob_params[0], c="purple")
    plt.xlabel("thickness 1 (km)")

    # plt.subplot(4, 2, 2)
    plt.subplot(3, 1, 2)
    plt.hist(vel_s[0, skip_inds:], bins=40, density=True)
    plt.axvline(bounds["vel_s"][0], c="black")
    plt.axvline(bounds["vel_s"][1], c="black")
    plt.axvline(m[1], c="red")
    plt.axvline(prob_params[1], c="purple")
    plt.xlabel("vel_s 1 (km/s)")

    # plt.subplot(4, 2, 3)
    plt.subplot(3, 1, 3)
    plt.hist(vel_s[1, skip_inds:], bins=40, density=True)
    plt.axvline(bounds["vel_s"][0], c="black")
    plt.axvline(bounds["vel_s"][1], c="black")
    plt.axvline(m[2], c="red")
    plt.axvline(prob_params[2], c="purple")
    plt.xlabel("vel_s 2 (km/s)")

    plt.tight_layout()
    plt.suptitle("MCMC model params")

    plt.show()


def plot_resulting_model(in_path):
    """
    plot the resulting model as velocity vs. depth
    with the histogram of probability for the thickness of the layer
    """

    fig = plt.figure()

    ds = xr.open_dataset(in_path)

    bounds = {
        "thickness": [0.001, 0.1],  # km
        "vel_s": [0.1, 1.8],  # km/s
    }

    _, _, prob_params = get_probable_model(in_path)

    t = prob_params[0]
    v1 = prob_params[1]
    v2 = prob_params[2]

    plt.clf()

    gs = GridSpec(4, 3, figure=fig)

    # Add subplots with custom spans
    ax1 = fig.add_subplot(gs[1:3, 0:2])

    # Add data to each subplot
    ax1.scatter([v1, v1, v2, v2], [0, t, t, 0.1])
    ax1.plot([v1, v1, v2, v2], [0, t, t, 0.1])

    ax1.text(
        v2 - 0.2,
        0.02,
        "t: "
        + str(np.round(t, 2))
        + " (km)\nv1: "
        + str(np.round(v1, 2))
        + " (km/s)\nv2: "
        + str(np.round(v2, 2))
        + " (km/s)",
    )

    ax1.set_xlim([0.1, 1.8])
    ax1.set_ylim([0, 0.1])

    # ax = plt.gca()
    # ax.set_ylim(ax.get_ylim()[::-1])
    plt.gca().invert_yaxis()

    ax1.set_xlabel("velocity (km/s)")
    ax1.set_ylabel("depth (km)")

    # plotting histograms
    model_params = ds["model_params"].values
    thickness = model_params[ds["thickness_inds"], :]
    vel_s = model_params[ds["vel_s_inds"], :]

    ax2 = fig.add_subplot(gs[0, 0:2])
    ax2.hist(vel_s[0, :], bins=40, density=True)
    ax2.axvline(bounds["vel_s"][0], c="black")
    ax2.axvline(bounds["vel_s"][1], c="black")
    # ax2.axvline(m[1], c="red")
    ax2.axvline(prob_params[1], c="purple")
    ax2.set_xlabel("vel_s 1 (km/s)")

    ax3 = fig.add_subplot(gs[3, 0:2])
    ax3.hist(vel_s[1, :], bins=40, density=True)
    ax3.axvline(bounds["vel_s"][0], c="black")
    ax3.axvline(bounds["vel_s"][1], c="black")
    # ax3.axvline(m[2], c="red")
    ax3.axvline(prob_params[2], c="purple")
    ax3.set_xlabel("vel_s 2 (km/s)")

    ax4 = fig.add_subplot(gs[1:3, 2])
    ax4.hist(
        thickness[0, :],
        bins=40,
        density=True,
        orientation="horizontal",
    )
    ax4.axvline(bounds["thickness"][0], c="black")
    ax4.axvline(bounds["thickness"][1], c="black")
    # ax4.axvline(m[0], c="red")
    ax4.axvline(prob_params[0], c="purple")
    ax4.set_xlabel("thickness 1 (km)")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def get_probable_model(in_path):
    ds = xr.open_dataset(in_path)

    # get the most probable model params
    # would add this to end of inversion later.
    model_params = ds["model_params"].values
    thickness = model_params[ds["thickness_inds"], :]
    vel_s = model_params[ds["vel_s_inds"], :]

    counts, bins, _ = plt.hist(thickness[0], bins=100, density=True)
    ind = np.argmax(counts)
    prob_thickness = (bins[ind] + bins[ind + 1]) / 2

    counts, bins, _ = plt.hist(vel_s[0], bins=100, density=True)
    ind = np.argmax(counts)
    prob_vel_s_0 = (bins[ind] + bins[ind + 1]) / 2

    counts, bins, _ = plt.hist(vel_s[1], bins=100, density=True)
    ind = np.argmax(counts)
    prob_vel_s_1 = (bins[ind] + bins[ind + 1]) / 2

    # need to change so can get vel_p and density when opening file.
    # or result model is saved at the end of run.
    sigma_model = {
        "thickness": 0.1,
        "vel_s": 0.1,
    }
    bounds = {
        "thickness": [0.001, 0.1],  # km
        "vel_s": [0.1, 1.8],  # km/s
    }
    model_params_kwargs = {
        "n_layers": 1,
        "sigma_model": sigma_model,
        "vpvs_ratio": 1.75,
        "param_bounds": bounds,
    }
    model_params = DispersionCurveParams(**model_params_kwargs)

    prob_params = np.array([prob_thickness, prob_vel_s_0, prob_vel_s_1])
    # run the forward model to predict data.
    velocity_model = model_params.get_velocity_model(prob_params)

    pd = PhaseDispersion(*velocity_model)
    pd_rayleigh = pd(ds["periods"].values, mode=0, wave="rayleigh")

    return pd_rayleigh.period, pd_rayleigh.velocity, prob_params


def plot_pred_vs_obs(in_path):
    """
    save data separately? in a separate .nc with true data if applicable.
    then the predicted model is read in from the inversion
    should the predicted model be saved directly, or computed from final model.
    """
    ds = xr.open_dataset(in_path)

    period, velocity, _ = get_probable_model(in_path)

    plt.clf()
    plt.plot(ds["period"], ds["data_true"], zorder=5)
    plt.scatter(ds["period"], ds["data_obs"], zorder=5)

    plt.plot(period, velocity, zorder=5)

    plt.plot(ds["period"], ds["data_pred"], c="grey", alpha=0.01, zorder=0)

    plt.xlabel("period")
    plt.ylabel("velocity")

    plt.legend(["data_obs", "data_true", "data_pred"])
    plt.show()


def plot_pred_hist(in_path):
    """
    save data separately? in a separate .nc with true data if applicable.
    then the predicted model is read in from the inversion
    should the predicted model be saved directly, or computed from final model.
    """
    ds = xr.open_dataset(in_path)

    # period, velocity, _ = get_probable_model(in_path)

    bins_list = []
    counts_list = []
    for ind, p in enumerate(ds["period"]):
        counts, bins, _ = plt.hist(ds["data_pred"][ind, :])
        bins_list.append(bins)
        counts_list.append(counts)

    plt.clf()
    plt.imshow(np.array(counts_list).T, aspect="auto")
    plt.xscale("log")
    plt.colorbar(norm="log")
    # plt.clf()
    # print(ds["period"].values.shape, ds["data_pred"].values.shape)
    # plt.imshow(ds["data_pred"].values, aspect="auto")

    plt.show()
