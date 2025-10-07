import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

    model_params_timeseries(
        input_ds,
        results_ds,
        save=True,
        out_filename=out_filename,
        plot_prob_model=plot_prob_model,
        plot_true_model=plot_true_model,
    )

    """
    save_inversion_info(input_ds, results_ds, out_filename=out_filename)

    model_params_timeseries(
        input_ds,
        results_ds,
        save=True,
        out_filename=out_filename,
        plot_prob_model=plot_prob_model,
        plot_true_model=plot_true_model,
    )
    model_params_autocorrelation(
        input_ds, results_ds, save=True, out_filename=out_filename
    )
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
    plot_likelihood(input_ds, results_ds, save=True, out_filename=out_filename)
    plot_covariance_matrix(input_ds, results_ds, save=True, out_filename=out_filename)
    plot_timestep_covariance_matrix(
        input_ds, results_ds, save=True, out_filename=out_filename
    )
    """


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

    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

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
    n_layers = int(input_ds.attrs["n_layers"])

    fig = make_subplots(n_layers + 1, n_param_types)  # , sharex=True, figsize=(14, 8)

    # loop over all params and plot
    # use param inds to get param name
    legend = []
    for c_ind, param in enumerate(param_types):
        inds = input_ds[param + "_inds"]
        bounds = input_ds["param_bounds"][inds]
        for r_ind, p in enumerate(model_params[inds]):
            legend.append(param + " " + str(r_ind + 1))
            # param timeseries
            fig.append_trace(go.Scatter(x=step, y=p), r_ind + 1, c_ind + 1)

            # true model
            if plot_true_model:
                fig.add_vline(true_model[inds][r_ind], r_ind + 1, c_ind + 1)
                # ax[r_ind, c_ind].axhline(true_model[inds][r_ind], c="red")
            # most probable model
            if plot_prob_model:
                fig.add_vline(probable_model[inds][r_ind], r_ind + 1, c_ind + 1)
                # ax[r_ind, c_ind].axhline(probable_model[inds][r_ind], c="purple")

            # bounds
            fig.update_yaxes(
                title_text=param + " " + str(r_ind + 1),
                range=[bounds[r_ind][0], bounds[r_ind][1]],
                row=r_ind + 1,
                col=c_ind + 1,
            )

            fig.update_xaxes(title_text="step", row=r_ind + 1, col=c_ind + 1)

    fig.update_layout(title_text="model params timeseries")

    if save:
        fig.write_html("figures/" + out_filename + "/time-" + out_filename + ".html")
    else:
        fig.show()


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

    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

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

    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

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
    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

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

    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

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
    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

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
    ax1.set_ylabel("depth (m)")

    ax1.set_xlim(ax1.get_xlim()[::-1])
    plt.gca().invert_yaxis()

    h = ax2.imshow(
        counts,
        norm=LogNorm(),
        extent=[vel_s_bins[0], vel_s_bins[-1], depth_bins[-1], depth_bins[0]],
        aspect="auto",
        interpolation="none",
    )

    # plot true model overtop
    if plot_true_model:
        true_model = np.array(true_model)
        ax2.plot(true_model[:, 1], true_model[:, 0], c="red")

    fig.colorbar(h, ax=ax2)
    ax2.set_xlabel("vel s (km/s)")

    # make these tick labels invisible
    ax2.tick_params("y", labelleft=False)

    plt.tight_layout()

    if save:
        plt.savefig("figures/" + out_filename + "/profile-" + out_filename + ".png")
    else:
        plt.show()


def plot_data_pred_histogram(
    input_ds, results_ds, n_bins=100, save=False, out_filename=""
):
    """
    plot all data predictions as a histogram.
    plot true data, observed data, and predicted data for the most probable model.
    """
    plt.clf()
    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    freqs = 1 / input_ds["period"]

    if "data_true" in input_ds:
        ax[0].plot(freqs, input_ds["data_true"], zorder=3, label="data_true")
    yerr = input_ds.attrs["sigma_data"]
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

    # flatten data_pred, repeat period
    hist_freqs = np.repeat(freqs, results_ds["data_pred"].shape[1])
    data_preds = results_ds["data_pred"].values.flatten()

    ax[0].hist2d(hist_freqs, data_preds, bins=n_bins, cmin=1, norm="log")
    # fig.colorbar(im, ax=ax, label="count")

    ax[0].set_xscale("log")
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("velocity (km/s)")

    ax[0].legend()

    ax[1].axhline(y=0, c="black")
    ax[1].scatter(
        freqs, results_ds["data_pred"].isel(step=pred_ind) - input_ds["data_obs"]
    )

    ax[1].set_xscale("log")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_title("residuals")

    if save:
        plt.savefig("figures/" + out_filename + "/data-" + out_filename + ".png")
    else:
        plt.show()


def plot_covariance_matrix(input_ds, results_ds, save=False, out_filename=""):
    """
    compare saved covariance matrix from sampling and covariance matrix from
    the final, full sample
    """
    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

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

    plt.clf()
    # cut results by step
    results_ds = results_ds.copy().isel(
        step=slice(input_ds.attrs["n_burn"], len(results_ds["step"]))
    )

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


def compare_results():
    # likelihood of most probable model for diff runs
    # BIC (number of parameters vs. likelihood of best model)
    pass
