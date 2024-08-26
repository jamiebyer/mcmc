import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import re
import xarray as xr


"""
TODO:
- will saving the output as a dask array keep the formatting of the arrays?
    right now the reading in takes extra steps to reformat...
"""


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

    # print(rot_mat_results)
    rot_mat_results = np.array([np.array(r) for r in rot_mat_results]).T

    # print(rot_mat_results)

    # print(rot_mat_results)
    # print(sigma_pd_results)

    # plt.plot(sigma_pd_results)
    # plt.show()
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


if __name__ == "__main__":
    out_dir = "./" + "out/inversion_results_1723861996.3231847"
    # plot_results()
    plot_zarr_file(out_dir=out_dir)

    # plot_params_hists(
    #    out_dir=out_dir
    # )

    # plot_ending_model()
