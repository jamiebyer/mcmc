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

# import statsmodels.api as sm


def plot_results(true_model):
    # plot true s and p velocities, plot observed phase velocity, plot density profile.
    plot_scene(true_model)

    # plot convergence...
    # plot PT swapping
    # plot likelihood against steps
    # plot model parameters over time
    # plot residuals of phase_vel_obs against steps
    # acceptance rate
    # step size...


def plot_scene(true_model):
    # *** get rid of repetition in plotting ***
    depths = np.cumsum(true_model.thickness) * 1000  # in m

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(12, 6))
    # plot true data
    data_phase_vel = true_model.phase_vel_true
    # data_depths = data_phase_vel / true_model.freqs  # *** check units ***
    # *** use the actual velocity ***
    data_depths = 3.5 / true_model.freqs  # *** check units ***
    ax1.plot(data_phase_vel, data_depths * 1000, label="true phase velocity")

    # plot observed data
    data_phase_vel = true_model.phase_vel_obs
    # data_depths = data_phase_vel / true_model.freqs  # *** check units ***
    data_depths = 3.5 / true_model.freqs  # *** check units ***
    # ax1.plot(data_phase_vel, data_depths * 1000, label="observed phase velocity")

    ax1.invert_yaxis()
    ax1.set_xlabel("km/s")
    ax1.ticklabel_format(style="scientific", scilimits=(-3, 3))
    ax1.set_ylabel("depth (m)")
    ax1.set_title("sigma_pd=" + str(true_model.sigma_pd))
    ax1.legend(loc="best")
    [ax1.axhline(y=d, c="black", alpha=0.25) for d in depths]

    # plot true model velocities
    ax2.plot(true_model.vel_s, depths, label="s velocity")
    ax2.plot(true_model.vel_p, depths, label="p velocity")

    ax2.ticklabel_format(style="scientific", scilimits=(-3, 3))
    ax2.set_xlabel("km/s")
    ax2.legend(loc="best")
    [ax2.axhline(y=d, c="black", alpha=0.25) for d in depths]

    # plot density depth profile
    ax3.plot(true_model.density, depths, label="density")
    ax3.ticklabel_format(style="scientific", scilimits=(-3, 3))
    ax3.set_xlabel("g/cm^3")
    ax3.legend(loc="best")
    [ax3.axhline(y=d, c="black", alpha=0.25) for d in depths]

    fig.tight_layout()
    # save to file
    fig.savefig("./figures/scene.png")


def plot_rotation_params(out_dir):
    inversion_results = pd.read_csv(out_dir)

    # *** want to read and write lists without doing this conversion. ***
    # *** or save parameters separately...
    rot_mat_results = inversion_results["rot_mat"]
    sigma_pd_results = inversion_results["sigma_pd"]

    # print(rot_mat_results)
    # print(rot_mat_results.str.lstrip("["))

    # print(rot_mat_results.str.lstrip("[").str.rstrip("]").str.split().head())

    # rot_mat_results = np.array(ast.literal_eval(s))
    """
    rot_mat_results = (
        rot_mat_results.str.lstrip("[")
        .str.rstrip("]")
        .str.split()
        .apply(lambda x: list(map(float, x)))
    )
    """

    rot_mat_results = (
        rot_mat_results.str.replace("   ", ",")
        .str.replace("  ", ",")
        .str.replace(" ", ",")
    )

    rot_mat_results = rot_mat_results.apply(ast.literal_eval)

    rot_mat_results = np.array([np.array(r) for r in rot_mat_results]).T

    rot_mag = [np.linalg.norm(mat) for mat in rot_mat_results]

    print(np.min(rot_mag), np.max(rot_mag))

    plt.plot(rot_mag)
    plt.show()


def plot_params_timeseries(out_dir):
    inversion_results = pd.read_csv(out_dir)

    # *** want to read and write lists without doing this conversion. ***
    # *** or save parameters separately...
    params_results = inversion_results["params"]
    params_results = (
        params_results.str.lstrip("[")
        .str.rstrip("]")
        .str.split()
        .apply(lambda x: list(map(float, x)))
        # .apply(ast.literal_eval)
    )
    params_results = np.array([np.array(p) for p in params_results]).T

    # figure directory
    figure_path = out_dir + "/figures"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    # create figures
    thickness = params_results[:10] * 1000  # convert from km to m
    vel_s = params_results[10:20]
    sigma_pd = params_results[-1]
    # plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))

    # plot timeseries for thickness parameters
    fig1 = plt.figure(figsize=(10, 6))
    # for row in range(rows):
    plt.plot(thickness.T)
    plt.legend(np.arange(len(thickness)))
    plt.title("thickness")
    plt.gca().ticklabel_format(style="sci", scilimits=(-3, 3))
    fig1.tight_layout()
    fig1.savefig(out_dir + "/figures/thickness_timeseries.png")

    # plot timeseries for shear velocity
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(vel_s.T)
    plt.legend(np.arange(len(vel_s)))
    plt.title("S velocity")
    plt.gca().ticklabel_format(style="sci", scilimits=(-3, 3))

    # fig2.suptitle("layer shear velocity (km/s)")
    fig2.tight_layout()
    fig2.savefig(out_dir + "/figures/vel_s_timeseries.png")

    # plot timeseries for sigma_pd
    fig3 = plt.figure()
    plt.plot(sigma_pd)
    plt.gca().ticklabel_format(style="sci", scilimits=(-3, 3))
    fig3.suptitle("sigma_pd")
    fig3.tight_layout()
    fig3.savefig(out_dir + "/figures/sigma_pd_timeseries.png")

    # plot logL
    fig4 = plt.figure()
    logL_results = inversion_results["logL"]
    logL_results.plot()
    plt.gca().ticklabel_format(style="sci", scilimits=(-3, 3))
    fig4.suptitle("logL")
    fig4.tight_layout()
    fig4.savefig(out_dir + "/figures/logL_timeseries.png")


def plot_params_hists(out_dir):
    inversion_results = pd.read_csv(out_dir)

    # *** want to read and write lists without doing this conversion. ***
    # *** or save parameters separately...
    params_results = inversion_results["params"]
    params_results = (
        params_results.str.lstrip("[")
        .str.rstrip("]")
        .str.split()
        .apply(lambda x: list(map(float, x)))
        # .apply(ast.literal_eval)
    )
    params_results = np.array([np.array(p) for p in params_results]).T

    # figure directory
    figure_path = out_dir + "/figures"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    # create figures
    thickness = params_results[:10] * 1000  # convert from km to m
    vel_s = params_results[10:20]
    sigma_pd = params_results[-1]

    # plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))

    rows, cols = 2, 5
    # plot histogram for thickness parameters
    fig1 = plt.figure(figsize=(10, 6))
    for row in range(rows):
        for col in range(cols):
            ind = (row * cols) + col
            plt.subplot(rows, cols, ind + 1)
            plt.hist(thickness[ind])
            plt.title("layer " + str(ind))
            plt.gca().ticklabel_format(style="sci", scilimits=(-3, 3))

    fig1.suptitle("layer thickness (m)")
    fig1.tight_layout()
    fig1.savefig(out_dir + "/figures/thickness_hist.png")

    # plot histogram for shear velocity
    fig2 = plt.figure(figsize=(10, 6))
    for row in range(rows):
        for col in range(cols):
            ind = (row * cols) + col
            plt.subplot(rows, cols, ind + 1)
            plt.hist(vel_s[ind])
            plt.title("layer " + str(ind))
            plt.gca().ticklabel_format(style="sci", scilimits=(-3, 3))

    fig2.suptitle("layer shear velocity (km/s)")
    fig2.tight_layout()
    fig2.savefig(out_dir + "/figures/vel_s_hist.png")

    # plot histogram for sigma_pd
    fig3 = plt.figure()
    plt.hist(sigma_pd)
    plt.gca().ticklabel_format(style="sci", scilimits=(-3, 3))
    fig3.suptitle("sigma_pd")
    fig3.tight_layout()
    fig3.savefig(out_dir + "/figures/sigma_pd_hist.png")

    # plot logL
    fig4 = plt.figure()
    logL_results = inversion_results["logL"]
    logL_results.hist()
    plt.gca().ticklabel_format(style="sci", scilimits=(-3, 3))
    fig4.suptitle("logL")
    fig4.tight_layout()
    fig4.savefig(out_dir + "/figures/logL_hist.png")


def plot_zarr_file(out_dir):
    ds = xr.open_zarr(out_dir)

    print(ds.step)

    plt.subplot(2, 1, 1)
    plt.imshow(ds.rot_mat.isel(step=0))
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(ds.rot_mat.isel(step=-1))
    plt.colorbar()

    # plt.plot(ds.params)
    plt.show()


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


def plot_starting_model():
    df = pd.read_csv("./results/inversion/optimize_model.csv", index_col=0)

    plt.plot(df["logL"])
    plt.ylabel("logL")
    plt.title("optimize model")
    plt.show()

    params = []
    for p in df["params"]:
        par = p.removeprefix("[").removesuffix("]")
        par = par.replace("'", "").replace("  ", " ")
        # par = par.split(" ")
        par = par.split(" ")
        par = [i for i in par if i]

        params.append(np.array(par, dtype=float))

    plt.subplot(3, 3, 1)
    plt.plot(np.array(params)[:, 0])
    plt.ylabel("thickness 1")

    plt.subplot(3, 3, 2)
    plt.plot(np.array(params)[:, 1])
    plt.ylabel("vel s 1")

    plt.subplot(3, 3, 3)
    plt.plot(np.array(params)[:, 2])
    plt.ylabel("vel s 2")

    plt.subplot(3, 3, 4)
    plt.plot(np.array(params)[:, 3])
    plt.ylabel("vel p 1")

    plt.subplot(3, 3, 5)
    plt.plot(np.array(params)[:, 4])
    plt.ylabel("vel p 2")

    plt.subplot(3, 3, 6)
    plt.plot(np.array(params)[:, 5])
    plt.ylabel("density 1")

    plt.subplot(3, 3, 7)
    plt.plot(np.array(params)[:, 6])
    plt.ylabel("density 2")

    plt.suptitle("optimize model")
    plt.show()


def plot_inversion_results_logL(in_path):
    ds = xr.open_dataset(in_path)

    plt.plot(ds["logL"][5:])
    plt.xlabel("step")
    plt.ylabel("logL")
    plt.title("MCMC model")
    plt.show()


def plot_inversion_results_param_time(in_path, skip_inds=0):
    ds = xr.open_dataset(in_path)

    bounds = {
        "thickness": [0.001, 0.1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 1.8],  # km/s
        "density": [0.5, 3],  # g/cm^3
    }

    m = [0.03] + [0.4, 1.5] + [1.6, 2.5] + [2.0, 2.5]

    # plt.subplot(4, 2, 1)
    plt.subplot(2, 2, 1)
    plt.scatter(
        np.arange(skip_inds, ds["thickness"].shape[0]),
        ds["thickness"][skip_inds:, 0],
        s=2,
    )
    plt.axhline(m[0], c="red")
    plt.axhline(bounds["thickness"][0], c="black")
    plt.axhline(bounds["thickness"][1], c="black")
    plt.ylabel("thickness 1 (km)")
    plt.xlabel("step")

    # plt.subplot(4, 2, 2)
    plt.subplot(2, 2, 2)
    plt.scatter(
        np.arange(skip_inds, ds["thickness"].shape[0]), ds["vel_s"][skip_inds:, 0], s=2
    )
    plt.axhline(m[1], c="red")
    plt.axhline(bounds["vel_s"][0], c="black")
    plt.axhline(bounds["vel_s"][1], c="black")
    plt.ylabel("vel_s 1 (km/s)")
    plt.xlabel("step")
    # plt.subplot(4, 2, 3)
    plt.subplot(2, 2, 3)
    plt.scatter(
        np.arange(skip_inds, ds["thickness"].shape[0]), ds["vel_s"][skip_inds:, 1], s=2
    )
    plt.axhline(m[2], c="red")
    plt.axhline(bounds["vel_s"][0], c="black")
    plt.axhline(bounds["vel_s"][1], c="black")
    plt.ylabel("vel_s 2 (km/s)")
    plt.xlabel("step")

    plt.subplot(2, 2, 4)
    plt.plot(np.arange(skip_inds, ds["thickness"].shape[0]), ds["acc_rate"][skip_inds:])
    plt.plot(
        np.arange(skip_inds, ds["thickness"].shape[0]), ds["err_ratio"][skip_inds:]
    )
    # plt.ylabel("")
    plt.legend(["acc_rate", "err_ratio"])

    """
    plt.subplot(4, 2, 4)
    plt.axhline(m[3], c="red")
    plt.plot(ds["vel_p"][:, 0])
    plt.axhline(bounds["vel_p"][0], c="black")
    plt.axhline(bounds["vel_p"][1], c="black")
    plt.ylabel("vel_p 1")
    plt.subplot(4, 2, 5)
    plt.plot(ds["vel_p"][:, 1])
    plt.axhline(m[4], c="red")
    plt.axhline(bounds["vel_p"][0], c="black")
    plt.axhline(bounds["vel_p"][1], c="black")
    plt.ylabel("vel_p 2")

    plt.subplot(4, 2, 6)
    plt.axhline(m[5], c="red")
    plt.plot(ds["density"][:, 0])
    plt.axhline(bounds["density"][0], c="black")
    plt.axhline(bounds["density"][1], c="black")
    plt.ylabel("density 1")
    plt.subplot(4, 2, 7)
    plt.plot(ds["density"][:, 1])
    plt.axhline(m[6], c="red")
    plt.axhline(bounds["density"][0], c="black")
    plt.axhline(bounds["density"][1], c="black")
    plt.ylabel("density 2")
    """
    # plt.subplot(3, 3, 9)
    # plt.axhline(bounds["sigma_model"][0], c="black")
    # plt.axhline(bounds["sigma_model"][1], c="black")
    # plt.title("sigma model")

    plt.tight_layout()
    plt.suptitle("MCMC model params")

    plt.show()


def plot_inversion_results_param_prob(in_path, skip_inds=0):
    ds = xr.open_dataset(in_path)

    m = [0.03] + [0.4, 1.5] + [1.6, 2.5] + [2.0, 2.5]
    # m = [0.5, 10.0, 7.00, 9.50, 3.50, 4.75, 2.00, 2.00]

    bounds = {
        "thickness": [0.001, 0.1],  # km
        "vel_p": [0.1, 6],  # km/s
        "vel_s": [0.1, 1.8],  # km/s
        "density": [0.5, 3],  # g/cm^3
    }

    # plt.plot(ds["logL"][5:])
    # plt.show()

    # plt.subplot(4, 2, 1)
    plt.subplot(3, 1, 1)
    plt.hist(ds["thickness"][skip_inds:, 0], bins=40, density=True)
    plt.axvline(bounds["thickness"][0], c="black")
    plt.axvline(bounds["thickness"][1], c="black")
    plt.axvline(m[0], c="red")
    plt.xlabel("thickness 1 (km)")

    # plt.subplot(4, 2, 2)
    plt.subplot(3, 1, 2)
    plt.hist(ds["vel_s"][skip_inds:, 0], bins=40, density=True)
    plt.axvline(bounds["vel_s"][0], c="black")
    plt.axvline(bounds["vel_s"][1], c="black")
    plt.axvline(m[1], c="red")
    plt.xlabel("vel_s 1 (km/s)")
    # plt.subplot(4, 2, 3)
    plt.subplot(3, 1, 3)
    plt.hist(ds["vel_s"][skip_inds:, 1], bins=40, density=True)
    plt.axvline(bounds["vel_s"][0], c="black")
    plt.axvline(bounds["vel_s"][1], c="black")
    plt.axvline(m[2], c="red")
    plt.xlabel("vel_s 2 (km/s)")

    """
    plt.subplot(4, 2, 4)
    plt.hist(ds["vel_p"][:, 0], bins=40)
    plt.axvline(bounds["vel_p"][0], c="black")
    plt.axvline(bounds["vel_p"][1], c="black")
    plt.axvline(m[3], c="red")
    plt.ylabel("vel_p 1")
    plt.subplot(4, 2, 5)
    plt.hist(ds["vel_p"][:, 1], bins=40)
    plt.axvline(bounds["vel_p"][0], c="black")
    plt.axvline(bounds["vel_p"][1], c="black")
    plt.axvline(m[4], c="red")
    plt.ylabel("vel_p 2")

    plt.subplot(4, 2, 6)
    plt.hist(ds["density"][:, 0], bins=40)
    plt.axvline(bounds["density"][0], c="black")
    plt.axvline(bounds["density"][1], c="black")
    plt.axvline(m[5], c="red")
    plt.ylabel("density 1")
    plt.subplot(4, 2, 7)
    plt.hist(ds["density"][:, 1], bins=40)
    plt.axvline(bounds["density"][0], c="black")
    plt.axvline(bounds["density"][1], c="black")
    plt.axvline(m[6], c="red")
    plt.ylabel("density 2")
    """

    plt.tight_layout()
    plt.suptitle("MCMC model params")

    plt.show()


def plot_pred_vs_obs(in_path):
    ds = xr.open_dataset(in_path)

    # determine the most probable most and use it to run the forward model

    plt.plot(ds["data_obs"])

    # plt.plot(ds.isel(step=slice(-20, -1))["data_pred"].T)

    pd = PhaseDispersion(*velocity_model)
    pd_rayleigh = pd(periods, mode=0, wave="rayleigh")

    plt.legend(["data_obs", "data_true", "data_pred"])
    plt.show()
