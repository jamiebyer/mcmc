import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import json
import os

from disba import PhaseDispersion
from disba import PhaseDispersion
from disba._exception import DispersionError


def plot_results(
    input_ds,
    results_ds,
    out_filename,
    plot_prob_model=False,
    plot_true_model=False,
):
    # make a folder for all plotting output
    if not os.path.isdir("./figures/" + out_filename):
        os.mkdir("./figures/" + out_filename)

    # plot_poster_results_data(
    #     input_ds, results_ds, n_bins=200, save=True, out_filename=out_filename
    # )
    # plot_poster_results_profile(
    #     input_ds, results_ds, n_bins=200, save=True, out_filename=out_filename
    # )
    # plot_data_pred_validate(input_ds, results_ds, save=True, out_filename=out_filename)

    # """
    save_inversion_info(input_ds, results_ds, out_filename=out_filename)

    model_params_timeseries(
        input_ds,
        results_ds,
        save=True,
        out_filename=out_filename,
        plot_prob_model=plot_prob_model,
        plot_true_model=plot_true_model,
    )
    # model_params_autocorrelation(
    #     input_ds, results_ds, save=True, out_filename=out_filename
    # )
    model_params_stepsize(input_ds, results_ds, save=True, out_filename=out_filename)
    model_params_histogram(
        input_ds,
        results_ds,
        save=True,
        out_filename=out_filename,
        plot_prob_model=plot_prob_model,
        plot_true_model=plot_true_model,
    )
    resulting_model_histogram(
        input_ds,
        results_ds,
        save=True,
        out_filename=out_filename,
        plot_prob_model=plot_prob_model,
        plot_true_model=plot_true_model,
    )
    plot_data_pred_histogram(input_ds, results_ds, save=True, out_filename=out_filename)
    # plot_data_pred_frequencies(
    #     input_ds, results_ds, save=True, out_filename=out_filename
    # )
    plot_likelihood(input_ds, results_ds, save=True, out_filename=out_filename)
    # plot_covariance_matrix(input_ds, results_ds, save=True, out_filename=out_filename)
    # plot_timestep_covariance_matrix(
    #     input_ds, results_ds, save=True, out_filename=out_filename
    # )
    # plot_vs30(input_ds, results_ds, save=True, out_filename=out_filename)
    # plot_surface_waves(input_ds, results_ds, save=True, out_filename=out_filename)
    # """


def save_inversion_info(input_ds, results_ds, out_filename=""):
    """
    everything should be saved already in input_ds and results_ds.
    this file shows the simple info in a text file so it's easy to look at with the results.

    save input model
    write input info to file

    - n_steps
    - length of computation
    - rotation
    - starting at true model
    - how much noise for synthetic data
    - field data vs. synthetic
    """

    output_dict = {
        # "param_bounds": input_ds["param_bounds"],
        "n_layers": int(input_ds.attrs["n_layers"]),
        "vpvs_ratio": float(input_ds.attrs["vpvs_ratio"]),
        "n_steps": int(np.max(results_ds["step"])),
        # "computation_time": float(results_ds.attrs["computation_time"]),
    }

    if "model_true" in input_ds:
        output_dict.update({"model_true": input_ds["model_true"].values.tolist()})

    # loop over param types and add bounds, proposal width
    depth_inds = input_ds["depth_inds"]
    vel_s_inds = input_ds["vel_s_inds"]
    output_dict.update(
        {
            "depth_bounds": input_ds["param_bounds"][depth_inds].values.tolist(),
            "vel_s_bounds": input_ds["param_bounds"][vel_s_inds].values.tolist(),
            "depth_width": input_ds["proposal_width"][depth_inds].values.tolist(),
            "vel_s_width": input_ds["proposal_width"][vel_s_inds].values.tolist(),
        }
    )

    json_str = json.dumps(output_dict, indent=4)
    with open("figures/" + out_filename + "/info-" + out_filename + ".json", "w") as f:
        f.write(json_str)


def model_params_timeseries(
    input_ds,
    results_ds,
    save=False,
    out_filename="",
    plot_prob_model=False,
    plot_true_model=False,
):
    """
    plot model params vs. time step.

    plot param bounds as black vertical lines.
    plot true model as red vertical lines.

    :param input_ds:
    :param results_ds:
    """
    # use input_ds to interpret results_ds
    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # use results_ds to get model params
    model_params = results_ds["model_params"].values
    step = results_ds["step"]

    # true model
    if plot_true_model:
        true_model = input_ds["model_true"]
    # get most probable model from ds_results
    if plot_prob_model:
        probable_model = results_ds["prob_params"]

    param_types = ["depth", "vel_s"]
    n_param_types = len(param_types)

    # one column for depth, one for vel_s, one for likelihood and acceptance
    n_layers = input_ds.attrs["n_layers"]

    fig, ax = plt.subplots(
        nrows=n_layers + 1, ncols=n_param_types, sharex=True, figsize=(14, 8)
    )

    # loop over all params and plot
    # use param inds to get param name
    legend = []
    for c_ind, param in enumerate(param_types):
        inds = input_ds[param + "_inds"]
        bounds = input_ds["param_bounds"][inds]
        for r_ind, p in enumerate(model_params[inds]):
            legend.append(param + " " + str(r_ind + 1))
            # param timeseries
            ax[r_ind, c_ind].scatter(
                step,
                p,
                s=2,
            )
            # true model
            if plot_true_model:
                ax[r_ind, c_ind].axhline(true_model[inds][r_ind], c="red")
            # most probable model
            if plot_prob_model:
                ax[r_ind, c_ind].axhline(probable_model[inds][r_ind], c="purple")
            # bounds
            ax[r_ind, c_ind].set_ylim([bounds[r_ind][0], bounds[r_ind][1]])

            # axis labels
            ax[r_ind, c_ind].set_ylabel(param + " " + str(r_ind + 1))
            ax[r_ind, c_ind].set_xlabel("step")

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.suptitle("model params timeseries")

    if save:
        plt.savefig("figures/" + out_filename + "/time-" + out_filename + ".png")
    else:
        plt.show()


def model_params_stepsize(
    input_ds,
    results_ds,
    save=False,
    out_filename="",
    plot_prob_model=False,
    plot_true_model=False,
):
    """
    plot model params vs. time step.

    plot param bounds as black vertical lines.
    plot true model as red vertical lines.

    :param input_ds:
    :param results_ds:
    """
    # use input_ds to interpret results_ds

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # use results_ds to get model params
    model_params = results_ds["model_params"].values
    step = results_ds["step"][:-1]

    params_diff = model_params[:, 1:] - model_params[:, :-1]

    # true model
    if plot_true_model:
        true_model = input_ds["model_true"]
    # get most probable model from ds_results
    if plot_prob_model:
        probable_model = results_ds["prob_params"]

    param_types = ["depth", "vel_s"]
    n_param_types = len(param_types)

    # one column for depth, one for vel_s, one for likelihood and acceptance
    n_layers = input_ds.attrs["n_layers"]

    fig, ax = plt.subplots(
        nrows=n_layers + 1, ncols=n_param_types, sharex=True, figsize=(14, 8)
    )

    # loop over all params and plot
    # use param inds to get param name
    legend = []
    for c_ind, param in enumerate(param_types):
        inds = input_ds[param + "_inds"]
        for r_ind, p in enumerate(params_diff[inds]):
            legend.append(param + " " + str(r_ind + 1))
            # param timeseries
            ax[r_ind, c_ind].scatter(
                step,
                p,
                s=2,
            )
            # true model
            if plot_true_model:
                ax[r_ind, c_ind].axhline(true_model[inds][r_ind], c="red")
            # most probable model
            if plot_prob_model:
                ax[r_ind, c_ind].axhline(probable_model[inds][r_ind], c="purple")

            # axis labels
            ax[r_ind, c_ind].set_ylabel(param + " " + str(r_ind + 1))
            ax[r_ind, c_ind].set_xlabel("step")

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.suptitle("model params step size")

    if save:
        plt.savefig("figures/" + out_filename + "/step-" + out_filename + ".png")
    else:
        plt.show()


def model_params_autocorrelation(
    input_ds,
    results_ds,
    save=False,
    out_filename="",
    plot_prob_model=False,
    plot_true_model=False,
):
    """
    plot model params vs. time step.

    plot param bounds as black vertical lines.
    plot true model as red vertical lines.

    :param input_ds:
    :param results_ds:
    """
    # use input_ds to interpret results_ds

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    param_types = ["depth", "vel_s"]
    n_param_types = len(param_types)

    # one column for depth, one for vel_s, one for likelihood and acceptance
    n_layers = input_ds.attrs["n_layers"]

    fig, ax = plt.subplots(
        nrows=n_layers + 1, ncols=n_param_types, sharex=True, figsize=(14, 8)
    )

    # loop over all params and plot
    # use param inds to get param name
    legend = []
    for c_ind, param in enumerate(param_types):
        inds = input_ds[param + "_inds"]
        for r_ind, p in enumerate(model_params[inds]):
            legend.append(param + " " + str(r_ind + 1))
            # param timeseries
            autocorr = np.correlate(p, p, mode="full")
            ax[r_ind, c_ind].plot(autocorr)

            # axis labels
            ax[r_ind, c_ind].set_ylabel(param + " " + str(r_ind + 1))
            # ax[r_ind, c_ind].set_xlabel("step")

    plt.suptitle("autocorrelation")
    plt.tight_layout()

    if save:
        plt.savefig("figures/" + out_filename + "/corr-" + out_filename + ".png")
    else:
        plt.show()


def plot_likelihood(input_ds, results_ds, save=False, out_filename=""):
    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # use results_ds to get model params
    step = results_ds["step"]

    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(step, results_ds["logL"])
    if "logL_true" in input_ds.attrs:
        plt.axhline(input_ds.attrs["logL_true"], c="red")
    plt.xlabel("step")
    plt.ylabel("logL")

    param_types = ["depth", "vel_s"]
    legend = []
    for c_ind, param in enumerate(param_types):
        inds = input_ds[param + "_inds"]
        for r_ind in range(np.sum(inds.values)):
            legend.append(param + " " + str(r_ind + 1))

    plt.subplot(3, 1, 2)
    plt.plot(step, results_ds["acc_rate"].T)

    plt.legend(legend)
    plt.xlabel("step")
    plt.ylabel("acceptance rate")

    plt.subplot(3, 1, 3)
    plt.plot(step, results_ds["bounds_err"].T)
    # plt.plot(step, results_ds["physics_err"].T)
    # plt.plot(step, results_ds["fm_err"].T)
    # plt.legend(["bounds error", "half-space error", "forward model error"])
    plt.xlabel("step")
    plt.ylabel("error ratio")

    plt.tight_layout()

    if save:
        plt.savefig("figures/" + out_filename + "/logL-" + out_filename + ".png")
    else:
        plt.show()


def model_params_histogram(
    input_ds,
    results_ds,
    n_bins=100,
    save=False,
    out_filename="",
    plot_prob_model=False,
    plot_true_model=False,
):
    """
    plot model params vs. time step.
    plot likelihood vs. time step.
    plot acceptance rate vs. time step.

    plot param bounds as black vertical lines.
    plot true model as red vertical lines.

    :param input_ds:
    :param results_ds:
    """
    # use input_ds to interpret results_ds
    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    # true model
    if plot_true_model:
        true_model = input_ds["model_true"]
    # get most probable model from ds_results
    if plot_prob_model:
        probable_model = results_ds["prob_params"]

    param_types = ["depth", "vel_s"]
    n_param_types = len(param_types)

    # one column for depth, one for vel_s
    n_layers = input_ds.attrs["n_layers"]
    fig, ax = plt.subplots(
        nrows=n_layers + 1,
        ncols=n_param_types,
        sharex="col",
        figsize=(14, 8),
    )

    # loop over all params and plot
    # use param inds to get param name
    for c_ind, param in enumerate(param_types):
        unit_scale = 1
        if param == "depth":
            unit_scale = 1000  # unit conversion to m
        inds = input_ds[param + "_inds"]
        bounds = unit_scale * input_ds["param_bounds"][inds]
        full_bounds = [np.min(bounds[:, 0]), np.max(bounds[:, 1])]

        for r_ind, p in enumerate(model_params[inds]):
            # full bounds
            ax[r_ind, c_ind].set_xlim([full_bounds[0], full_bounds[1]])
            # param bounds
            ax[r_ind, c_ind].axvspan(
                bounds[r_ind][0], bounds[r_ind][1], color="blue", alpha=0.1
            )
            ax[r_ind, c_ind].axvline(bounds[r_ind][0], c="black")
            ax[r_ind, c_ind].axvline(bounds[r_ind][1], c="black")

            bins = np.linspace(
                bounds[r_ind][0],
                bounds[r_ind][1],
                n_bins,
            )
            # param timeseries
            ax[r_ind, c_ind].hist(unit_scale * p, bins=bins, density=True)

            # true model
            if plot_true_model:
                ax[r_ind, c_ind].axvline(unit_scale * true_model[inds][r_ind], c="red")

            # most probable model
            if plot_prob_model:
                ax[r_ind, c_ind].axvline(
                    unit_scale * probable_model[inds][r_ind], c="purple"
                )

            # axis labels
            ax[r_ind, c_ind].set_xlabel(param + " " + str(r_ind + 1))

    plt.tight_layout()

    if save:
        plt.savefig("figures/" + out_filename + "/hist-" + out_filename + ".png")
    else:
        plt.show()


def model_params_histogram_compare(
    input_ds_list,
    results_ds_list,
    n_bins=100,
    save=False,
    # out_filename="",
    plot_prob_model=False,
    plot_true_model=False,
):
    """
    plot model params vs. time step.
    plot likelihood vs. time step.
    plot acceptance rate vs. time step.

    plot param bounds as black vertical lines.
    plot true model as red vertical lines.

    :param input_ds:
    :param results_ds:
    """

    n_layers = input_ds_list[0].attrs["n_layers"]

    param_types = ["depth", "vel_s"]
    n_param_types = len(param_types)

    fig, ax = plt.subplots(
        nrows=n_layers + 1,
        ncols=n_param_types,
        sharex="col",
        figsize=(14, 8),
    )

    for ind in range(len(input_ds_list)):
        input_ds = input_ds_list[ind]
        results_ds = results_ds_list[ind]

        # use input_ds to interpret results_ds
        # n_burn = input_ds.attrs["n_burn"]
        n_burn = int(len(results_ds["step"]) / 3)

        # cut results by step
        results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

        # use results_ds to get model params
        model_params = results_ds["model_params"].values

        # true model
        if plot_true_model:
            true_model = input_ds["model_true"]
        # get most probable model from ds_results
        if plot_prob_model:
            probable_model = results_ds["prob_params"]

        # one column for depth, one for vel_s
        n_layers = input_ds.attrs["n_layers"]

        # loop over all params and plot
        # use param inds to get param name
        for c_ind, param in enumerate(param_types):
            unit_scale = 1
            if param == "depth":
                unit_scale = 1000  # unit conversion to m
            inds = input_ds[param + "_inds"]
            bounds = unit_scale * input_ds["param_bounds"][inds]
            full_bounds = [np.min(bounds[:, 0]), np.max(bounds[:, 1])]

            for r_ind, p in enumerate(model_params[inds]):
                # full bounds
                ax[r_ind, c_ind].set_xlim([full_bounds[0], full_bounds[1]])
                # param bounds
                # ax[r_ind, c_ind].axvspan(
                #     bounds[r_ind][0], bounds[r_ind][1], color="blue", alpha=0.1
                # )
                ax[r_ind, c_ind].axvline(bounds[r_ind][0], c="black")
                ax[r_ind, c_ind].axvline(bounds[r_ind][1], c="black")

                bins = np.linspace(
                    bounds[r_ind][0],
                    bounds[r_ind][1],
                    n_bins,
                )
                # param timeseries
                ax[r_ind, c_ind].hist(
                    unit_scale * p, bins=bins, density=True, histtype="step"
                )

                # true model
                if plot_true_model:
                    ax[r_ind, c_ind].axvline(
                        unit_scale * true_model[inds][r_ind], c="red"
                    )

                # most probable model
                if plot_prob_model:
                    ax[r_ind, c_ind].axvline(
                        unit_scale * probable_model[inds][r_ind], c="purple"
                    )

                # axis labels
                ax[r_ind, c_ind].set_xlabel(param + " " + str(r_ind + 1))

    plt.tight_layout()

    if save:
        plt.savefig("figures/hist_compare.png")
    else:
        plt.show()


def resulting_model_histogram(
    input_ds,
    results_ds,
    n_bins=100,
    save=False,
    out_filename="",
    plot_prob_model=False,
    plot_true_model=False,
):
    """
    plot the resulting model as velocity vs. depth
    with the histogram of probability for the depth of the layer
    """

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    # true model
    if plot_true_model:
        true_params = input_ds["model_true"].values

    # define hist bins between bounds
    # use param inds to get depth, and use min and max of all depth bounds
    depth_bounds = input_ds["param_bounds"][input_ds["depth_inds"]]
    depth_bins = (
        np.linspace(
            np.min(depth_bounds[:, 0]),
            np.max(depth_bounds[:, 1]),
            n_bins,
        )
        * 1000
    )  # unit conversion
    vel_s_bounds = input_ds["param_bounds"][input_ds["vel_s_inds"]]
    vel_s_bins = np.linspace(
        np.min(vel_s_bounds[:, 0]), np.max(vel_s_bounds[:, 1]), n_bins
    )
    counts = np.zeros((n_bins, n_bins))

    # loop over every resulting model
    # add vel_s 1 to hist bins above depth
    # add vel_s 2 to hist bins below depth

    depth_inds = input_ds["depth_inds"]
    vel_s_inds = input_ds["vel_s_inds"]

    n_steps = len(results_ds["step"])

    depth = model_params[depth_inds] * 1000  # unit conversion to m
    depth_plotting = np.concatenate(
        (
            np.zeros((1, n_steps)),
            depth,
            np.full((1, n_steps), np.max(depth_bounds[:, 1])) * 1000,  # unit conversion
        ),
        axis=0,
    )
    vel_s = model_params[vel_s_inds]

    # for each layer
    # for each sample / step
    for layer_ind in range(input_ds.attrs["n_layers"] + 1):
        for step_ind in range(n_steps):
            # find bin index closest to layer depth
            depth_upper_inds = np.argmin(
                abs(depth_bins - depth_plotting[layer_ind, step_ind])
            )
            depth_lower_inds = np.argmin(
                abs(depth_bins - depth_plotting[layer_ind + 1, step_ind])
            )
            # find bin index closest to layer vel_s
            vel_s_close_inds = np.argmin(abs(vel_s_bins - vel_s[layer_ind, step_ind]))

            counts[depth_upper_inds:depth_lower_inds, vel_s_close_inds] += 1

    # plot true model overtop
    if plot_true_model:
        true_depth = true_params[depth_inds] * 1000
        true_vel_s = true_params[vel_s_inds]
        true_depth_plotting = np.concatenate(
            ([0], true_depth, [np.max(depth_bounds[:, 1]) * 1000])
        )

        true_model = []
        for layer_ind in range(input_ds.attrs["n_layers"] + 1):
            true_model.append([true_depth_plotting[layer_ind], true_vel_s[layer_ind]])
            true_model.append(
                [true_depth_plotting[layer_ind + 1], true_vel_s[layer_ind]]
            )

    fig = plt.figure()
    gs = GridSpec(1, 3, figure=fig)

    # Add subplots with custom spans
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:], sharey=ax1)

    # plot depth histogram
    for ind in range(input_ds.attrs["n_layers"]):
        ax1.hist(
            depth[ind],
            bins=depth_bins,
            density=True,
            orientation="horizontal",
        )

    ax1.set_ylim(
        [
            np.min(depth_bounds[:, 0]) * 1000,
            np.max(depth_bounds[:, 1]) * 1000,
        ]
    )
    ax1.set_ylabel("Depth (m)")

    ax1.set_xlim(ax1.get_xlim()[::-1])
    plt.gca().invert_yaxis()

    counts[counts == 0] = np.nan
    h = ax2.imshow(
        counts,
        extent=[vel_s_bins[0], vel_s_bins[-1], depth_bins[-1], depth_bins[0]],
        aspect="auto",
        interpolation="none",
    )

    # plot true model overtop
    if plot_true_model:
        true_model = np.array(true_model)
        ax2.plot(true_model[:, 1], true_model[:, 0], c="white")
        ax2.plot(true_model[:, 1], true_model[:, 0],
            c="black",
            ls=(0, (5, 5)),
            label="true model",
        )

    fig.colorbar(h, ax=ax2)
    ax2.set_xlabel("Shear-wave velocity (km/s)")

    # make these tick labels invisible
    ax2.tick_params("y", labelleft=False)

    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig("figures/" + out_filename + "/profile-" + out_filename + ".png")
    else:
        plt.show()


def get_noise_params(noise_dist, input_ds, inv):
    if inv:
        prefix = "inv_"
    else:
        prefix = ""
    
    noise_params = {}
    noise_params["frequency_scaling"] = (input_ds.attrs[prefix + "frequency_scaling"] == 1)
    if noise_dist == "normal":
        noise_params["std"] = input_ds.attrs[prefix + "std"]
    elif noise_dist == "asym-laplace":
        noise_params["lambd"] = input_ds.attrs[prefix + "lambd"]
        noise_params["kappa"] = input_ds.attrs[prefix + "kappa"]
        noise_params["lambd_scale"] = input_ds.attrs[prefix + "lambd_scale"]
    
    return noise_params

def get_pdf(noise_dist, noise_params):
    x = np.linspace(-25, 25, 100000)
    mu = 0
    
    if noise_dist == "normal":
        std_data = noise_params["std"]
        std_data = np.mean(std_data)

        if isinstance(std_data, float):
            std = std_data
        else:
            std = std_data[freq_ind]

        pdf = (1 / np.sqrt(2 * np.pi * std**2)) * np.exp(
            -((x - mu) ** 2 / (2 * std**2))
        )

    if noise_dist == "asym-laplace":
        lambd, kappa = noise_params["lambd"], noise_params["kappa"]
        lambd_scaling = noise_params["lambd_scale"]

        lambd = (1 / lambd_scaling) * lambd

        if isinstance(lambd, float):
            l = lambd
        else:
            l = lambd[freq_ind]

        s = np.sign(x - mu)
        pdf = (l / (kappa + 1 / kappa)) * np.exp(-(x - mu) * l * s * kappa**s)
    
    return x, pdf


def plot_data_pred_histogram(
    input_ds, results_ds, n_bins=100, save=False, out_filename=""
):
    """
    plot all data predictions as a histogram.
    plot true data, observed data, and predicted data for the most probable model.
    """

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    plt.clf()
    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 14))
    freqs = 1 / input_ds["period"]

    if "data_true" in input_ds:
        ax[0, 0].plot(freqs, input_ds["data_true"], c="white")
        ax[0, 0].plot(freqs, input_ds["data_true"],
            c="black",
            ls=(0, (5, 5)),
            label="data_true",
        )

    # yerr = input_ds.attrs["noise_percent"]
    yerr = None
    ax[0, 0].errorbar(
        freqs,
        input_ds["data_obs"],
        yerr,
        fmt="o",
        zorder=3,
        c="orange",
        label="data_obs",
    )

    # get data prediction
    pred_ind = np.argmax(results_ds["logL"].values)
    # ax[0, 0].scatter(
    #     freqs, results_ds["data_pred"].isel(step=pred_ind), zorder=3, label="data_pred"
    # )
    ax[0, 0].plot(
        freqs, results_ds["data_pred"].isel(step=pred_ind), zorder=3, label="data_pred", c="red"
    )
    # estimated error
    # *** depends if it's a percent error or not
    # yerr = input_ds.attrs["sigma_data"] * results_ds["data_prob"]

    # flatten data_pred, repeat period
    hist_freqs = np.repeat(freqs, results_ds["data_pred"].shape[1])
    data_preds = results_ds["data_pred"].values.flatten()

    # make log spaced freq bin sizes
    freq_bins = np.logspace(
        np.log10(np.min(freqs)), np.log10(np.max(freqs)), len(freqs) + 1
    )
    data_bins = np.linspace(np.min(data_preds), np.max(data_preds), n_bins)

    ax[0, 0].hist2d(hist_freqs, data_preds, bins=[freq_bins, data_bins], cmin=1)
    # fig.colorbar(im, ax=ax, label="count")

    # ax[0, 0].set_ylim([0, 1.0])
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("frequency (Hz)")
    ax[0, 0].set_ylabel("velocity (km/s)")

    ax[0, 0].legend()

    ax[0, 1].axhline(y=0, c="black")
    residuals = (
        input_ds["data_obs"] - results_ds["data_pred"].isel(step=pred_ind)
    )  #  / input_ds.attrs["noise_percent"]
    ax[0, 1].scatter(freqs, residuals)

    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("frequency (Hz)")
    ax[0, 1].set_ylabel(
        # "standardized residuals\n(data pred - data obs) / noise percent"
        "residuals\n(data obs - data pred)"
    )

    # plot residuals as histogram
    ax[1, 1].hist(residuals, bins=16)

    # plot pdf overtop
    noise_dist = input_ds.attrs["noise_dist"]
    inv_noise_dist = input_ds.attrs["inv_noise_dist"]
    noise_params = get_noise_params(noise_dist, input_ds, inv=False)
    inv_noise_params = get_noise_params(inv_noise_dist, input_ds, inv=True)
    x, pdf = get_pdf(noise_dist, noise_params)
    inv_x, inv_pdf = get_pdf(inv_noise_dist, inv_noise_params)

    ax[1, 1].plot(x, pdf, label="noise dist")
    ax[1, 1].plot(inv_x, inv_pdf, label="model noise dist")
    
    ax[1, 1].axvline(x=0, c="black")
    ax[1, 1].set_xlabel(
        # "standardized residuals\n(data pred - data obs) / noise percent"
        "residuals\n(data pred - data obs)"
    )
    ax[1, 1].set_ylabel("counts")
    ax[1, 1].set_xlim([-0.5, 0.5])
    ax[1, 1].legend()

    # plot data predictions - data obs
    # print(results_ds["data_pred"].shape)
    # print(input_ds["data_obs"].shape)
    diff = (results_ds["data_pred"] - input_ds["data_true"]).values.flatten()
    diff_bins = np.linspace(np.min(diff), np.max(diff), n_bins)

    # ax[1, 0].hist2d(hist_freqs, diff, bins=[freq_bins, diff_bins], cmin=1, norm="log")
    ax[1, 0].hist2d(hist_freqs, diff, bins=[freq_bins, diff_bins], cmin=1)
    ax[1, 0].axhline(y=0, c="black")
    # fig.colorbar(im, ax=ax, label="count")

    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("frequency (Hz)")
    ax[1, 0].set_ylabel("data pred - data true")

    if save:
        plt.savefig("figures/" + out_filename + "/data-" + out_filename + ".png")
    else:
        plt.show()


def plot_data_pred_frequencies(
    input_ds, results_ds, n_bins=100, save=False, out_filename=""
):
    """
    plot all data predictions as a histogram.
    plot true data, observed data, and predicted data for the most probable model.
    """

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    pred_ind = np.argmax(results_ds["logL"].values)

    noise_percent = input_ds.attrs["noise_percent"]
    if input_ds.attrs["noise_dist"] == "asym-laplace":
        lambd, kappa = input_ds.attrs["lambd"], input_ds.attrs["kappa"]
        lambd = (1 / noise_percent) * lambd

    x = np.linspace(-50, 50, 100000)

    if not os.path.exists("figures/" + out_filename + "/data-freqs"):
        os.makedirs("figures/" + out_filename + "/data-freqs/")

    for ind, p in enumerate(input_ds["period"].values):
        mu = input_ds["data_obs"][ind].values

        f = 1 / p
        plt.clf()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

        # plot error distribution
        if input_ds.attrs["noise_dist"] == "normal":
            pdf = (1 / np.sqrt(2 * np.pi * noise_percent[ind] ** 2)) * np.exp(
                -((x - mu) ** 2 / (2 * noise_percent[ind] ** 2))
            )
        elif input_ds.attrs["noise_dist"] == "asym-laplace":
            s = np.sign(x - mu)
            pdf = (lambd[ind] / (kappa + 1 / kappa)) * np.exp(
                -(x - mu) * lambd[ind] * s * kappa**s
            )

        # flatten data_pred, repeat period
        # data_preds = results_ds["data_pred"].isel(period=ind).values
        data_preds = results_ds["data_pred"].values[ind, :]

        ax.hist(data_preds, bins=n_bins, density=True)
        # fig.colorbar(im, ax=ax, label="count")

        ax.plot(x, pdf, c="black")

        # plot true data as a vertical line
        if "data_true" in input_ds:
            ax.axvline(input_ds["data_true"][ind], label="data_true", c="red")

        ax.axvline(input_ds["data_obs"][ind], label="data_obs", c="orange")
        ax.axvline(
            results_ds["data_pred"].isel(step=pred_ind)[ind],
            label="data_pred",
            c="purple",
        )

        # ax[0].set_xscale("log")
        # ax[0].set_xlabel("frequency (Hz)")
        # ax[0].set_ylabel("velocity (km/s)")

        # ax[0].legend()
        if input_ds.attrs["noise_dist"] == "asym-laplace":
            ax.set_title(
                "freq: " + str(np.round(f, 2)) + "\nlambda: " + str(lambd[ind])
            )
        else:
            ax.set_title("freq: " + str(np.round(f, 2)))
        ax.set_xlim([0, 2])

        if save:
            plt.savefig(
                "figures/"
                + out_filename
                + "/data-freqs/"
                + str(np.round(f, 2))
                + out_filename
                + ".png"
            )
        else:
            plt.show()


def plot_data_pred_validate(
    input_ds, results_ds, n_bins=200, save=False, out_filename=""
):
    """
    plot all data predictions as a histogram.
    plot true data, observed data, and predicted data for the most probable model.
    """

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    plt.clf()
    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    freqs = 1 / input_ds["period"]

    """
    if "data_true" in input_ds:
        ax[0].plot(freqs, input_ds["data_true"], zorder=3, label="data_true")
        ax[1].plot(freqs, input_ds["data_true"], zorder=3, label="data_true")

    # yerr = input_ds.attrs["noise_percent"]
    yerr = None
    ax[0].errorbar(
        freqs,
        input_ds["data_obs"],
        yerr,
        fmt="o",
        zorder=3,
        c="orange",
        label="data_obs",
    )

    # get data prediction
    pred_ind = np.argmax(results_ds["logL"].values)
    ax[0].scatter(
        freqs, results_ds["data_pred"].isel(step=pred_ind), zorder=3, label="data_pred"
    )
    # estimated error
    # *** depends if it's a percent error or not
    # yerr = input_ds.attrs["sigma_data"] * results_ds["data_prob"]
    """
    # flatten data_pred, repeat period
    hist_freqs = np.repeat(freqs, results_ds["data_pred"].shape[1])
    data_preds = results_ds["data_pred"].values.flatten()

    # make log spaced freq bin sizes
    freq_bins = np.logspace(
        np.log10(np.min(freqs)), np.log10(np.max(freqs)), len(freqs) + 1
    )
    data_bins = np.linspace(np.min(data_preds), np.max(data_preds), n_bins)

    ax[0].hist2d(hist_freqs, data_preds, bins=[freq_bins, data_bins], cmin=1)
    # fig.colorbar(im, ax=ax, label="count")

    ax[0].set_ylim([0.15, 0.80])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("velocity (km/s)")

    ax[0].legend()

    # functions needed for forward model
    def get_vel_p(vel_s):
        vel_p = vel_s * 1.75
        return vel_p

    def get_density(vel_p):
        # using Garner's relation
        density = (1741 * np.sign(vel_p) * abs(vel_p) ** (1 / 4)) / 1000
        return density

    def forward_model(periods, model_params, depth_inds, vel_s_inds):
        """
        get phase dispersion curve for current shear velocities and layer thicknesses.

        :param periods:
        :param velocity model: velocity model for disba has the format
            [thickness (km), vel_p (km/s), vel_s (km/s), density (g/cm3)]
        :param model_params: model params to use to get phase dispersion
        """
        depth = model_params[depth_inds]
        vel_s = model_params[vel_s_inds]
        # get thicknesses
        # *** probably a faster way to do this
        depth = np.concatenate(([0], depth))
        thickness = np.concatenate((depth[1:] - depth[:-1], [0]))

        # assemble params into velocity model
        vel_p = get_vel_p(vel_s)
        density = get_density(vel_p)
        # avoid converting thickness back and forth from list
        velocity_model = np.array([thickness, vel_p, vel_s, density])

        # phase dispersion object
        pd = PhaseDispersion(*velocity_model)

        # try calculating phase_velocity from given params.
        try:
            pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
            phase_velocity = pd_rayleigh.velocity
            return phase_velocity
        except (DispersionError, ZeroDivisionError) as e:
            raise e

    # plot data pred from model params
    periods = input_ds["period"].values
    # periods_ = periods
    periods_ = np.flip(1 / np.logspace(-2, 1.5, 100))
    freqs_ = 1 / periods_
    # read in model params
    model_params = results_ds["model_params"].values
    depth_inds = input_ds["depth_inds"]
    vel_s_inds = input_ds["vel_s_inds"]

    data_preds, hist_freqs = [], []
    for i in range(model_params.shape[1]):
        data_pred = forward_model(periods_, model_params[:, i], depth_inds, vel_s_inds)
        data_preds.append(data_pred)
        hist_freqs += list(freqs_)

        ax[1].plot(freqs_, data_pred, c="black", alpha=0.3)
    data_preds = np.array(data_preds).flatten()

    print(model_params[:, :20000][:, ind_min])
    print(model_params[:, :20000][:, ind_max])

    data_preds, hist_freqs = [], []
    # for i in range(model_params.shape[1]):
    # for i in range(20000):
    for i in [ind_min, ind_max]:
        data_pred = forward_model(periods, model_params[:, i], depth_inds, vel_s_inds)
        data_preds.append(data_pred)
        hist_freqs += list(freqs)

        ax[1].plot(freqs, data_pred)
    data_preds = np.array(data_preds).flatten()
    
    ax[1].errorbar(
        freqs,
        input_ds["data_obs"],
        yerr,
        fmt="o",
        zorder=3,
        c="orange",
        label="data_obs",
    )

    # get data prediction
    pred_ind = np.argmax(results_ds["logL"].values)
    ax[1].scatter(
        freqs, results_ds["data_pred"].isel(step=pred_ind), zorder=3, label="data_pred"
    )
    # estimated error
    # *** depends if it's a percent error or not
    # yerr = input_ds.attrs["sigma_data"] * results_ds["data_prob"]
    # make log spaced freq bin sizes
    freq_bins = np.logspace(
        np.log10(np.min(freqs)), np.log10(np.max(freqs)), len(freqs) + 1
    )
    data_bins = np.linspace(np.min(data_preds), np.max(data_preds), n_bins)

    # ax[1].hist2d(hist_freqs, data_preds, bins=[freq_bins, data_bins], cmin=1)
    # fig.colorbar(im, ax=ax, label="count")

    ax[1].set_ylim([0.1, 1.2])
    ax[1].set_xscale("log")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("velocity (km/s)")

    ax[1].legend()

    if save:
        plt.savefig(
            "figures/"
            + out_filename
            + "/data-validate-line-ex-"
            + out_filename
            + ".png"
        )
    else:
        plt.show()


def plot_data_pred_validate_v2(
    input_ds, results_ds, n_bins=200, save=False, out_filename=""
):
    """
    plot all data predictions as a histogram.
    plot true data, observed data, and predicted data for the most probable model.
    """

    """
    [0.05216616 0.09721406 0.20081375 0.99307072 1.15611619]
    [0.05087318 0.11241771 0.20108244 0.63974797 1.99930884]
    
    inversion test model 4
    [0.05193434 0.14719832 0.20144648 0.84446044 1.48015709]
    [0.18545995 0.1854601  0.18545995 0.1854601  0.18545995 0.1854601
    0.18545995 0.1854601  0.18546025 0.1854601  0.18546025 0.18546041
    0.18546056 0.18546102 0.18546117 0.18546193 0.18546239 0.18546315
    0.18546422 0.18546559 0.18546758 0.18546987 0.18547246 0.18547628
    0.1854807  0.18548604 0.18549291 0.1855013  0.18551122 0.18552358
    0.18553838 0.18555593 0.18557714 0.18560231 0.18563207 0.18566701
    0.18570836 0.18575643 0.18581243 0.18587728 0.1859525  0.18603932
    0.18613927 0.18625356 0.18638463 0.18653401 0.18670415 0.18689748
    0.18711705 0.18736531 0.18764592 0.18796315 0.18832067 0.18872335
    0.18917699 0.18968709 0.19026097 0.19090657 0.19163274 0.19244954
    0.19336949 0.19440633 0.19557653 0.19689931 0.19839818 0.20010091
    0.20204198 0.2042629  0.20681645 0.2097701  0.2132105  0.2172527
    0.22205266 0.22782888 0.23489812 0.24374074 0.25512395 0.27036397
    0.29190923 0.32471105 0.37648625 0.40415502 0.41186894 0.41877782
    0.42600713 0.43383153 0.44241399 0.45190282 0.46245915 0.47427159
    0.48756719 0.50262303 0.51978398 0.53948033 0.56225164 0.58878026
    0.6199149  0.65669164 0.70028844 0.75181981]


    [thickness (km), vel_p (km/s), vel_s (km/s), density (g/cm3)]
    
    [[0.0508385  0.05521109 0.        ]
    [0.35320441 1.10252875 2.88043206]
    [0.20183109 0.63001643 1.64596118]
    [1.34216312 1.78400576 2.26810513]]

    [[0.05131082 0.06046549 0.        ]
    [0.3498575  1.29315574 2.72616689]
    [0.19991857 0.73894614 1.55780965]
    [1.33897222 1.85657065 2.23710764]]
    
    [[0.05263225 0.03770616 0.        ]
    [0.35289606 1.4900028  2.40992638]
    [0.20165489 0.85143017 1.37710079]
    [1.34187009 1.92351484 2.1692006 ]]
    """

    # functions needed for forward model
    def get_vel_p(vel_s):
        vel_p = vel_s * 1.75
        return vel_p

    def get_density(vel_p):
        # using Garner's relation
        density = (1741 * np.sign(vel_p) * abs(vel_p) ** (1 / 4)) / 1000
        return density

    def forward_model(periods, model_params, depth_inds, vel_s_inds):
        """
        get phase dispersion curve for current shear velocities and layer thicknesses.

        :param periods:
        :param velocity model: velocity model for disba has the format
            [thickness (km), vel_p (km/s), vel_s (km/s), density (g/cm3)]
        :param model_params: model params to use to get phase dispersion
        """
        depth = model_params[depth_inds]
        vel_s = model_params[vel_s_inds]
        # get thicknesses
        # *** probably a faster way to do this
        depth = np.concatenate(([0], depth))
        thickness = np.concatenate((depth[1:] - depth[:-1], [0]))

        # assemble params into velocity model
        vel_p = get_vel_p(vel_s)
        density = get_density(vel_p)
        # avoid converting thickness back and forth from list
        velocity_model = np.array([thickness, vel_p, vel_s, density])
        # print(velocity_model)

        # phase dispersion object
        pd = PhaseDispersion(*velocity_model)

        # try calculating phase_velocity from given params.
        try:
            pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
            phase_velocity = pd_rayleigh.velocity
            return phase_velocity
        except (DispersionError, ZeroDivisionError) as e:
            raise e

    depth_inds = np.array([True, True, False, False, False])
    vel_s_inds = np.array([False, False, True, True, True])

    # freqs = np.logspace(-2, 1.2)
    periods = np.flip(1 / np.logspace(0, 1.1, 100))

    m1 = np.array([0.0508385, 0.10604959, 0.20183109, 0.63001643, 1.64596118])
    dc1 = np.array(
        [
            0.18581404,
            0.18581419,
            0.18581404,
            0.18581419,
            0.18581404,
            0.18581419,
            0.18581404,
            0.18581419,
            0.18581434,
            0.18581449,
            0.18581465,
            0.1858148,
            0.18581495,
            0.18581541,
            0.18581587,
            0.18581632,
            0.18581709,
            0.18581816,
            0.18581953,
            0.18582121,
            0.1858235,
            0.18582609,
            0.1858296,
            0.18583372,
            0.18583906,
            0.18584562,
            0.1858534,
            0.18586302,
            0.18587507,
            0.18588896,
            0.1859062,
            0.18592649,
            0.18595045,
            0.18597898,
            0.1860127,
            0.18605223,
            0.18609846,
            0.18615232,
            0.18621473,
            0.18628721,
            0.18637037,
            0.18646635,
            0.18657636,
            0.18670194,
            0.18684553,
            0.18700895,
            0.18719435,
            0.18740476,
            0.18764265,
            0.18791166,
            0.18821485,
            0.1885568,
            0.18894148,
            0.18937437,
            0.18986036,
            0.19040678,
            0.19102003,
            0.19170835,
            0.19248152,
            0.1933502,
            0.19432661,
            0.19542509,
            0.19666303,
            0.19806028,
            0.19964033,
            0.20143217,
            0.20347029,
            0.2057971,
            0.20846601,
            0.21154386,
            0.21511701,
            0.21929838,
            0.22423902,
            0.23014829,
            0.23732405,
            0.24620786,
            0.25748945,
            0.27231779,
            0.29277418,
            0.32304487,
            0.37110792,
            0.40515302,
            0.41390577,
            0.42115888,
            0.42859876,
            0.43656965,
            0.4452461,
            0.45477583,
            0.46531446,
            0.47704023,
            0.49016981,
            0.50497175,
            0.52178419,
            0.54104292,
            0.56331617,
            0.58936079,
            0.62020124,
            0.65724287,
            0.70242414,
            0.75836469,
        ]
    )

    m2 = np.array([0.05131082, 0.11177631, 0.19991857, 0.73894614, 1.55780965])
    dc2 = np.array(
        [
            0.18405327,
            0.18405343,
            0.18405327,
            0.18405343,
            0.18405327,
            0.18405343,
            0.18405327,
            0.18405343,
            0.18405358,
            0.18405343,
            0.18405358,
            0.18405373,
            0.18405419,
            0.18405434,
            0.1840548,
            0.18405526,
            0.18405572,
            0.18405678,
            0.18405785,
            0.18405923,
            0.1840609,
            0.1840635,
            0.1840664,
            0.18406991,
            0.18407433,
            0.18407998,
            0.18408684,
            0.18409554,
            0.18410577,
            0.18411812,
            0.18413323,
            0.18415108,
            0.1841726,
            0.18419808,
            0.18422814,
            0.18426369,
            0.18430535,
            0.18435372,
            0.18441033,
            0.18447579,
            0.18455163,
            0.18463906,
            0.18473962,
            0.18485451,
            0.1849862,
            0.1851365,
            0.18530724,
            0.18550149,
            0.18572137,
            0.18597024,
            0.18625176,
            0.18656899,
            0.18692712,
            0.18732979,
            0.18778344,
            0.18829323,
            0.18886651,
            0.18951089,
            0.19023522,
            0.19104958,
            0.19196618,
            0.19299843,
            0.19416283,
            0.19547768,
            0.19696648,
            0.19865609,
            0.20057976,
            0.2027781,
            0.20530205,
            0.20821633,
            0.21160393,
            0.21557473,
            0.22027611,
            0.22591347,
            0.23278252,
            0.2413279,
            0.25225609,
            0.26677376,
            0.28715783,
            0.31831292,
            0.37145898,
            0.40300164,
            0.41025536,
            0.41701531,
            0.42416833,
            0.43193108,
            0.44044152,
            0.44983697,
            0.46026757,
            0.4719146,
            0.4849984,
            0.49979363,
            0.51664635,
            0.53600335,
            0.55844323,
            0.58473443,
            0.61590203,
            0.6533294,
            0.69885369,
            0.75480279,
        ]
    )

    m3 = np.array(
        [
            0.05263225,
            0.09033841,
            0.20165489,
            0.85143017,
            1.37710079,
        ]
    )
    dc3 = np.array(
        [
            0.18565186,
            0.18565171,
            0.18565186,
            0.18565171,
            0.18565186,
            0.18565201,
            0.18565186,
            0.18565201,
            0.18565186,
            0.18565201,
            0.18565216,
            0.18565232,
            0.18565247,
            0.18565262,
            0.18565308,
            0.18565354,
            0.18565399,
            0.18565476,
            0.18565552,
            0.18565689,
            0.18565857,
            0.18566056,
            0.18566315,
            0.18566635,
            0.18567017,
            0.1856752,
            0.18568146,
            0.18568894,
            0.18569824,
            0.18570938,
            0.18572296,
            0.18573899,
            0.18575867,
            0.18578171,
            0.18580933,
            0.18584183,
            0.18587982,
            0.18592453,
            0.18597687,
            0.18603745,
            0.18610749,
            0.18618881,
            0.18628235,
            0.18638962,
            0.18651245,
            0.18665299,
            0.18681336,
            0.18699539,
            0.18720215,
            0.18743637,
            0.18770172,
            0.18800125,
            0.18833924,
            0.18872025,
            0.18914948,
            0.18963242,
            0.19017578,
            0.1907872,
            0.19147461,
            0.19224808,
            0.1931189,
            0.19409989,
            0.19520661,
            0.19645706,
            0.19787262,
            0.19947922,
            0.2013086,
            0.2033989,
            0.20579865,
            0.20856919,
            0.21178925,
            0.2155632,
            0.22003296,
            0.22539566,
            0.23193909,
            0.24010056,
            0.25059143,
            0.26467118,
            0.28486298,
            0.31731507,
            0.38265167,
            0.40848938,
            0.41438263,
            0.42069824,
            0.42758881,
            0.4351532,
            0.44349823,
            0.45274719,
            0.46305207,
            0.47459656,
            0.48761017,
            0.50238281,
            0.51928192,
            0.5387854,
            0.56151398,
            0.58829102,
            0.62020325,
            0.65866944,
            0.70543518,
            0.76226319,
        ]
    )

    models = [m1, m2, m3]
    dcs = [dc1, dc2, dc3]
    colours = ["red", "orange", "blue"]
    """
    for i in range(len(models)):
        data_pred = forward_model(periods, models[i], depth_inds, vel_s_inds)
        plt.plot(periods, data_pred, c=colours[i])
        plt.plot(periods, dcs[i], c=colours[i], ls="-")
    """

    data_pred = forward_model(periods, m1, depth_inds, vel_s_inds)
    plt.plot(1 / periods, 1 / (data_pred * 1000))
    plt.plot(1 / periods, 1 / (dc1 * 1000))

    # ax[1].set_ylim([0.15, 0.80])
    plt.xscale("log")
    plt.xlabel("frequency (Hz)")
    # plt.ylabel("velocity (km/s)")
    plt.ylabel("slowness (s/m)")

    if save:
        plt.savefig(
            "figures/" + out_filename + "/data-validate2-" + out_filename + ".png"
        )
    else:
        plt.show()


def plot_covariance_matrix(input_ds, results_ds, save=False, out_filename=""):
    """
    compare saved covariance matrix from sampling and covariance matrix from
    the final, full sample
    """
    n_burn = input_ds.attrs["n_burn"]
    # n_burn = int(len(results_ds["step"])/3)

    plt.clf()
    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    # normalize model params for computing cov mat
    # model_params_norm = np.empty(model_params.shape)
    model_params_norm = np.full(model_params.shape, np.nan)

    param_names = ["depth", "vel_s"]
    bounds = input_ds["param_bounds"].values
    # for s in range(len(results_ds["step"])):
    for s in range(model_params.shape[1]):
        model_params_norm[:, s] = (model_params[:, s] - bounds[:, 0]) / bounds[:, 2]

    mean_diff = (model_params_norm.T - np.mean(model_params_norm, axis=1)).T
    cov_mat_final = (1 / mean_diff.shape[1]) * mean_diff @ mean_diff.T
    # cov_mat_final = np.cov(model_params_norm)

    computed_cov_mat = results_ds["cov_mat"][:, :, -1]

    percent_diff = np.abs(cov_mat_final - computed_cov_mat) / (
        np.abs(cov_mat_final - computed_cov_mat) / 2
    )

    n_model_params = input_ds.coords["n_model_params"]

    corr_final = np.full((len(n_model_params), len(n_model_params)), np.nan)
    corr_computed = np.full((len(n_model_params), len(n_model_params)), np.nan)

    for i in n_model_params:
        for j in n_model_params:
            corr_final[i, j] = cov_mat_final[i, j] / np.sqrt(
                cov_mat_final[i, i] * cov_mat_final[j, j]
            )
            corr_computed[i, j] = computed_cov_mat[i, j] / np.sqrt(
                computed_cov_mat[i, i] * computed_cov_mat[j, j]
            )

    corr_percent_diff = np.abs(corr_final - corr_computed) / (
        (corr_final + corr_computed) / 2
    )

    param_names = []
    param_types = ["depth", "vel_s"]
    for p in param_types:
        inds = input_ds[p + "_inds"]
        for i in range(np.sum(inds.values)):
            param_names.append(p[0] + " " + str(i + 1))

    # plt.clf()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    # computed_cov_mat[computed_cov_mat == 0.0] = np.nan
    im = ax[0, 0].imshow(computed_cov_mat, interpolation="none")
    ax[0, 0].set_title("cov mat from inversion run")
    plt.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_xticks(n_model_params, param_names)
    ax[0, 0].set_yticks(n_model_params, param_names)

    im = ax[0, 1].imshow(cov_mat_final)
    ax[0, 1].set_title("cov mat computed after run")
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_xticks(n_model_params, param_names)
    ax[0, 1].set_yticks(n_model_params, param_names)

    # percent difference
    im = ax[0, 2].imshow(percent_diff)
    ax[0, 2].set_title("percent difference")
    plt.colorbar(im)
    ax[0, 2].set_xticks(n_model_params, param_names)
    ax[0, 2].set_yticks(n_model_params, param_names)

    # correlation matrix
    im = ax[1, 0].imshow(corr_computed)
    ax[1, 0].set_title("corr mat from inversion run")
    plt.colorbar(im)
    ax[1, 0].set_xticks(n_model_params, param_names)
    ax[1, 0].set_yticks(n_model_params, param_names)

    im = ax[1, 1].imshow(corr_final)
    ax[1, 1].set_title("corr mat computed after run")
    plt.colorbar(im)
    ax[1, 1].set_xticks(n_model_params, param_names)
    ax[1, 1].set_yticks(n_model_params, param_names)

    # percent difference
    im = ax[1, 2].imshow(corr_percent_diff)
    ax[1, 2].set_title("percent difference")
    plt.colorbar(im)
    ax[1, 2].set_xticks(n_model_params, param_names)
    ax[1, 2].set_yticks(n_model_params, param_names)

    if save:
        plt.savefig("figures/" + out_filename + "/cov-" + out_filename + ".png")
    else:
        plt.show()


def plot_timestep_covariance_matrix(input_ds, results_ds, save=False, out_filename=""):
    """
    plot correlation / covariance matrix for different timesteps
    """

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    plt.clf()
    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    timesteps = [3000, 8000, 15000]

    cov_mat_1 = results_ds["cov_mat"][:, :, timesteps[0]]
    cov_mat_2 = results_ds["cov_mat"][:, :, timesteps[1]]
    cov_mat_3 = results_ds["cov_mat"][:, :, timesteps[2]]

    step_size_1 = results_ds["proposal_width"][:, timesteps[0]]
    step_size_2 = results_ds["proposal_width"][:, timesteps[1]]
    step_size_3 = results_ds["proposal_width"][:, timesteps[2]]

    n_model_params = input_ds.coords["n_model_params"]
    corr_mat_1 = np.empty((len(n_model_params), len(n_model_params)))
    corr_mat_2 = np.empty((len(n_model_params), len(n_model_params)))
    corr_mat_3 = np.empty((len(n_model_params), len(n_model_params)))

    for i in n_model_params:
        for j in n_model_params:
            corr_mat_1[i, j] = cov_mat_1[i, j] / np.sqrt(
                cov_mat_1[i, i] * cov_mat_1[j, j]
            )
            corr_mat_2[i, j] = cov_mat_2[i, j] / np.sqrt(
                cov_mat_2[i, i] * cov_mat_2[j, j]
            )
            corr_mat_3[i, j] = cov_mat_3[i, j] / np.sqrt(
                cov_mat_3[i, i] * cov_mat_3[j, j]
            )

    param_names = []
    param_types = ["depth", "vel_s"]
    for p in param_types:
        inds = input_ds[p + "_inds"]
        for i in range(np.sum(inds.values)):
            param_names.append(p[0] + " " + str(i + 1))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    im = ax[0, 0].imshow(corr_mat_1)
    ax[0, 0].set_title("corr mat " + str(timesteps[0]))
    plt.colorbar(im)
    ax[0, 0].set_xticks(n_model_params, param_names)
    ax[0, 0].set_yticks(n_model_params, param_names)

    im = ax[0, 1].imshow(corr_mat_2)
    ax[0, 1].set_title("corr mat " + str(timesteps[1]))
    plt.colorbar(im)
    ax[0, 1].set_xticks(n_model_params, param_names)
    ax[0, 1].set_yticks(n_model_params, param_names)

    # percent difference
    im = ax[1, 0].imshow(corr_mat_3)
    ax[1, 0].set_title("corr mat " + str(timesteps[2]))
    plt.colorbar(im)
    ax[1, 0].set_xticks(n_model_params, param_names)
    ax[1, 0].set_yticks(n_model_params, param_names)

    # correlation matrix
    ax[1, 1].plot(step_size_1)
    ax[1, 1].plot(step_size_2)
    ax[1, 1].plot(step_size_3)
    ax[1, 1].set_title("step size")

    if save:
        plt.savefig("figures/" + out_filename + "/corr-time-" + out_filename + ".png")
    else:
        plt.show()


def compare_rotation():
    """
    compare timeseries for un-rotated and rotated run.
    """
    pass


def compare_n_layers():
    # likelihood of most probable model for diff runs
    # BIC (number of parameters vs. likelihood of best model)
    pass


def plot_vs30(
    input_ds,
    results_ds,
    save=False,
    out_filename="",
):
    """
    Vs30 = sum(d_i)/sum(t_i) = 30/sum(d_i/v_i)

    Description	VS30 range (m/s)
    Hard rock	1500
    Rock	760-1500
    Very dense soil and soft rock	360-760
    Stiff soil	180-360
    Soil with soft clay	<180
    Site-specific analysis required	---
    """
    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    depth_bounds = input_ds["param_bounds"][input_ds["depth_inds"]]

    depth_inds = input_ds["depth_inds"]
    vel_s_inds = input_ds["vel_s_inds"]

    n_steps = len(results_ds["step"])

    depth = model_params[depth_inds] * 1000  # unit conversion to m
    depth_plotting = np.concatenate(
        (
            np.zeros((1, n_steps)),
            depth,
            np.full((1, n_steps), np.max(depth_bounds[:, 1])) * 1000,  # unit conversion
        ),
        axis=0,
    )

    vel_s = model_params[vel_s_inds]

    depth_boundary = 30
    Vs30_list = []
    # for each layer
    # for each sample / step
    for step_ind in range(n_steps):
        # find first depth after 30 m
        depth_diff = depth_plotting[:, step_ind] - depth_boundary
        depth_diff[depth_diff < 0] = np.inf

        # smallest positive number
        layer_ind = np.argmin(depth_diff)
        depth_plotting[layer_ind] = 30

        thickness = (
            depth_plotting[1 : layer_ind + 1, step_ind]
            - depth_plotting[:layer_ind, step_ind]
        )

        Vs30 = 30 / np.sum(thickness[: layer_ind + 1] / vel_s[:layer_ind, step_ind])
        Vs30_list.append(Vs30)

    fig = plt.figure()

    plt.hist(np.array(Vs30_list) * 1000, bins=100, density=True)

    classes = [
        ["A", "Hard\nrock", 1500, 1550],
        ["B", "Rock", 760, 900],
        ["C", "Very dense\nsoil and soft rock", 360, 380],
        ["D", "Stiff\nsoil", 180, 230],
        ["E", "Soil with\nsoft clay", 0, 0],
    ]
    for c, name, vert, loc in classes:
        plt.text(loc, 0.11, name)
        plt.axvline(vert, c="k", ls="-")

    # add percentage for classification

    plt.tight_layout()

    if save:
        plt.savefig("figures/" + out_filename + "/vs30-" + out_filename + ".png")
    else:
        plt.show()


def plot_surface_waves(input_ds, results_ds, n_bins=100, save=False, out_filename=""):
    """
    Look at some resulting models from the inversion.
    Use disba to plot Rayleigh waves and Love waves.

    data pred type plot for Love waves?
    """

    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

    periods = input_ds["period"].values
    freqs = 1 / periods

    # use results_ds to get model params
    model_params = results_ds["model_params"].values.T

    depth_inds = input_ds["depth_inds"]
    vel_s_inds = input_ds["vel_s_inds"]

    vpvs_ratio = input_ds.attrs["vpvs_ratio"]

    pd_rayleigh_list, pd_love_list = [], []
    # loop over each model
    for m in model_params:
        # get rayleigh and love dispersion curves for each model
        depth = m[depth_inds]
        vel_s = m[vel_s_inds]

        # get thicknesses
        depth = np.concatenate(([0], depth))
        thickness = np.concatenate((depth[1:] - depth[:-1], [0]))

        vel_p = vel_s * vpvs_ratio
        density = (1741 * np.sign(vel_p) * abs(vel_p) ** (1 / 4)) / 1000
        # avoid converting thickness back and forth from list
        velocity_model = np.array([thickness, vel_p, vel_s, density])

        # phase dispersion object
        pd = PhaseDispersion(*velocity_model)

        pd_rayleigh = pd(periods, mode=0, wave="rayleigh")
        pd_love = pd(periods, mode=0, wave="love")

        pd_rayleigh_list.append(pd_rayleigh.velocity)
        pd_love_list.append(pd_love.velocity)

    # plotting
    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

    # flatten data_pred, repeat period
    hist_freqs = np.repeat(freqs, model_params.shape[0])
    # data_preds = results_ds["data_pred"].values.flatten()

    pd_rayleigh_list = np.array(pd_rayleigh_list).T.flatten()
    pd_love_list = np.array(pd_love_list).T.flatten()
    print(freqs.shape, hist_freqs.shape, pd_rayleigh_list.shape, pd_love_list.shape)
    ax[0].hist2d(hist_freqs, pd_rayleigh_list, bins=n_bins, cmin=1, norm="log")
    ax[1].hist2d(hist_freqs, pd_love_list, bins=n_bins, cmin=1, norm="log")
    # fig.colorbar(im, ax=ax, label="count")

    ax[0].set_xscale("log")
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("velocity (km/s)")

    ax[1].set_xscale("log")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("velocity (km/s)")

    if save:
        plt.savefig(
            "figures/" + out_filename + "/surface-waves-" + out_filename + ".png"
        )
    else:
        plt.show()


def plot_poster_results_data(
    input_ds, results_ds, n_bins=100, save=False, out_filename=""
):
    plt.clf()
    # fig = plt.figure(figsize=(10, 18))

    # ax = fig.add_subplot()  # data pred hist
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    freqs = 1 / input_ds["period"]

    ax.plot(freqs, input_ds["data_true"], zorder=3, c="white", linewidth=3)
    ax.plot(
        freqs,
        input_ds["data_true"],
        zorder=3,
        label="true data",
        c="black",
        ls=(0, (5, 5)),
        linewidth=3,
    )

    # yerr = input_ds.attrs["noise_percent"]
    # yerr = None
    ax.scatter(
        freqs,
        input_ds["data_obs"],
        # fmt="o",
        zorder=3,
        c="white",
        edgecolor="black",
        label="observed data",
    )

    # get data prediction
    # pred_ind = np.argmax(results_ds["logL"].values)
    # ax1.scatter(
    #     freqs, results_ds["data_pred"].isel(step=pred_ind), zorder=3, label="data_pred"
    # )
    # estimated error
    # *** depends if it's a percent error or not
    # yerr = input_ds.attrs["sigma_data"] * results_ds["data_prob"]

    # flatten data_pred, repeat period
    hist_freqs = np.repeat(freqs, results_ds["data_pred"].shape[1])
    data_preds = results_ds["data_pred"].values.flatten()

    # make log spaced freq bin sizes
    freq_bins = np.logspace(
        np.log10(np.min(freqs)), np.log10(np.max(freqs)), len(freqs) + 1
    )
    data_bins = np.linspace(np.min(data_preds), np.max(data_preds), n_bins)

    h = ax.hist2d(hist_freqs, data_preds, bins=[freq_bins, data_bins], cmin=1)
    # fig.colorbar(im, ax=ax, label="count")

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_ylim([0.2, 1.0])
    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)", fontsize=20)
    ax.set_ylabel("phase velocity (km/s)", fontsize=20)

    cb = fig.colorbar(h[3], ax=ax)
    cb.set_ticks([])

    ax.legend(fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)

    if save:
        plt.savefig("figures/" + out_filename + "/poster-data-" + out_filename + ".png")
    else:
        plt.show()


def plot_poster_results_profile(
    input_ds, results_ds, n_bins=100, save=False, out_filename=""
):
    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig)

    # Add subplots with custom spans
    ax1 = fig.add_subplot(gs[0:2, 1:])  # depth profile
    ax2 = fig.add_subplot(gs[0:2, 0], sharey=ax1)  # depth marginal
    ax3 = fig.add_subplot(gs[2, 1:], sharex=ax1)  # vel s marginals

    # n_burn = input_ds.attrs["n_burn"]
    n_burn = int(len(results_ds["step"]) / 3)

    # cut results by step
    results_ds = results_ds.copy().isel(step=slice(n_burn, len(results_ds["step"])))

    # PROFILE

    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    # true model
    true_params = input_ds["model_true"].values

    # define hist bins between bounds
    # use param inds to get depth, and use min and max of all depth bounds
    depth_bounds = input_ds["param_bounds"][input_ds["depth_inds"]]
    depth_bins = (
        np.linspace(
            np.min(depth_bounds[:, 0]),
            np.max(depth_bounds[:, 1]),
            n_bins,
        )
        * 1000
    )  # unit conversion
    vel_s_bounds = input_ds["param_bounds"][input_ds["vel_s_inds"]]
    vel_s_bins = np.linspace(
        np.min(vel_s_bounds[:, 0]), np.max(vel_s_bounds[:, 1]), n_bins
    )
    counts = np.zeros((n_bins, n_bins))

    # loop over every resulting model
    # add vel_s 1 to hist bins above depth
    # add vel_s 2 to hist bins below depth

    depth_inds = input_ds["depth_inds"]
    vel_s_inds = input_ds["vel_s_inds"]

    n_steps = len(results_ds["step"])

    depth = model_params[depth_inds] * 1000  # unit conversion to m
    depth_plotting = np.concatenate(
        (
            np.zeros((1, n_steps)),
            depth,
            np.full((1, n_steps), np.max(depth_bounds[:, 1])) * 1000,  # unit conversion
        ),
        axis=0,
    )
    vel_s = model_params[vel_s_inds]

    # for each layer
    # for each sample / step
    for layer_ind in range(input_ds.attrs["n_layers"] + 1):
        for step_ind in range(n_steps):
            # find bin index closest to layer depth
            depth_upper_inds = np.argmin(
                abs(depth_bins - depth_plotting[layer_ind, step_ind])
            )
            depth_lower_inds = np.argmin(
                abs(depth_bins - depth_plotting[layer_ind + 1, step_ind])
            )
            # find bin index closest to layer vel_s
            vel_s_close_inds = np.argmin(abs(vel_s_bins - vel_s[layer_ind, step_ind]))

            counts[depth_upper_inds:depth_lower_inds, vel_s_close_inds] += 1

    # plot true model overtop
    true_depth = true_params[depth_inds] * 1000
    true_vel_s = true_params[vel_s_inds]
    true_depth_plotting = np.concatenate(
        ([0], true_depth, [np.max(depth_bounds[:, 1]) * 1000])
    )

    true_model = []
    for layer_ind in range(input_ds.attrs["n_layers"] + 1):
        true_model.append([true_depth_plotting[layer_ind], true_vel_s[layer_ind]])
        true_model.append([true_depth_plotting[layer_ind + 1], true_vel_s[layer_ind]])

    counts[counts == 0] = np.nan
    h = ax1.imshow(
        counts,
        extent=[vel_s_bins[0], vel_s_bins[-1], depth_bins[-1], depth_bins[0]],
        aspect="auto",
        interpolation="none",
    )

    # plot true model overtop
    true_model = np.array(true_model)
    ax1.plot(true_model[:, 1], true_model[:, 0], c="white", linewidth=3)
    ax1.plot(
        true_model[:, 1],
        true_model[:, 0],
        c="black",
        ls=(0, (5, 5)),
        label="true model",
        linewidth=3,
    )
    ax1.legend(fontsize=18)

    cb = fig.colorbar(h, ax=ax1)
    cb.set_ticks([])
    # cb.set_label("counts", fontsize=20)
    # cb.ax.tick_params(labelsize=16)
    # ax2.set_xlabel("vel s (km/s)")

    # make these tick labels invisible
    ax1.tick_params("y", labelleft=False)

    # plot depth histogram
    for ind in range(input_ds.attrs["n_layers"]):
        ax2.hist(
            depth[ind],
            bins=depth_bins,
            density=True,
            orientation="horizontal",
            color="darkgrey",
        )
        # ax2.axhline(true_depth, c="red")
        ax2.axhline(true_depth, c="white", linewidth=3)
        ax2.axhline(true_depth, c="black", ls=(0, (5, 5)), linewidth=3)

    ax2.set_ylim(
        [
            np.min(depth_bounds[:, 0]) * 1000,
            np.max(depth_bounds[:, 1]) * 1000,
        ]
    )
    ax2.set_ylabel("depth (m)", fontsize=20)

    ax2.set_xlim(ax2.get_xlim()[::-1])
    ax2.set_ylim(ax2.get_ylim()[::-1])

    # plt.gca().invert_yaxis()

    # plot vel s histogram
    for ind in range(input_ds.attrs["n_layers"] + 1):
        if ind == 0:
            c = "darkgrey"
        else:
            c = "dimgrey"
        ax3.hist(
            vel_s[ind],
            bins=vel_s_bins,
            density=True,
            color=c,
            label="Vs " + str(ind + 1),
        )
        # ax3.axvline(true_vel_s[ind], c="red")
        ax3.axvline(true_vel_s[ind], c="white", linewidth=3)
        ax3.axvline(true_vel_s[ind], c="black", ls=(0, (5, 5)), linewidth=3)

    ax3.legend(fontsize=18)

    ax3.set_xlim([0.2, 1.2])
    ax3.set_xlabel("shear velocity (km/s)", fontsize=20)

    ax1.tick_params(axis="both", which="major", labelsize=18)
    ax2.tick_params(axis="both", which="major", labelsize=18)
    ax3.tick_params(axis="both", which="major", labelsize=18)

    # plt.tight_layout()

    if save:
        plt.savefig(
            "figures/" + out_filename + "/poster-profile-" + out_filename + ".png"
        )
    else:
        plt.show()
