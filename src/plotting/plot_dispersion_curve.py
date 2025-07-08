import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec


def model_params_timeseries(input_ds, results_ds):
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

    # use results_ds to get model params
    model_params = results_ds["model_params"].values
    step = results_ds["step"]

    # true model
    true_model = input_ds["model_true"]
    # get most probable model from ds_results
    probable_model = results_ds["prob_params"]

    param_types = ["thickness", "vel_s"]
    n_param_types = len(param_types)

    # one column for thickness, one for vel_s, one for likelihood and acceptance
    # n_layers = input_ds["n_layers"]
    n_layers = 2
    fig, ax = plt.subplots(nrows=n_layers + 1, ncols=n_param_types + 1)

    # loop over all params and plot
    # use param inds to get param name
    legend = []
    for c_ind, param in enumerate(param_types):
        inds = input_ds[param + "_inds"]
        for r_ind, p in enumerate(model_params[inds]):
            legend.append(param + " " + str(r_ind))
            # param timeseries
            ax[r_ind, c_ind].scatter(
                step,
                p,
                s=2,
            )
            # true model
            ax[r_ind, c_ind].axhline(true_model[inds][r_ind], c="red")
            # most probable model
            ax[r_ind, c_ind].axhline(probable_model[inds][r_ind], c="purple")
            # bounds
            ax[r_ind, c_ind].axhline(input_ds.attrs[param + "_bounds"][0], c="black")
            ax[r_ind, c_ind].axhline(input_ds.attrs[param + "_bounds"][1], c="black")

            # axis labels
            ax[r_ind, c_ind].set_ylabel(param + " " + str(r_ind + 1))
            ax[r_ind, c_ind].set_xlabel("step")

    # plot likelihood in last column
    ax[0, -1].plot(step, results_ds["logL"])
    ax[0, -1].set_xlabel("step")
    ax[0, -1].set_ylabel("logL")

    # acceptance rate
    ax[1, -1].plot(step, results_ds["acc_rate"].T)
    ax[1, -1].legend(legend)
    ax[1, -1].set_xlabel("step")
    ax[1, -1].set_ylabel("acceptance rate")

    # error rate
    ax[2, -1].plot(step, results_ds["err_ratio"].T)
    ax[2, -1].legend(legend)
    ax[2, -1].set_xlabel("step")
    ax[2, -1].set_ylabel("error rate")

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.show()


def model_params_histogram(input_ds, results_ds, n_bins=100):
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

    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    # true model
    true_model = input_ds["model_true"]
    # get most probable model from ds_results
    probable_model = results_ds["prob_params"]

    param_types = ["thickness", "vel_s"]
    n_param_types = len(param_types)

    # one column for thickness, one for vel_s, one for likelihood and acceptance
    # n_layers = input_ds["n_layers"]
    n_layers = 2
    fig, ax = plt.subplots(nrows=n_layers + 1, ncols=n_param_types)

    # loop over all params and plot
    # use param inds to get param name
    for c_ind, param in enumerate(param_types):
        inds = input_ds[param + "_inds"]
        bins = np.linspace(
            input_ds.attrs[param + "_bounds"][0],
            input_ds.attrs[param + "_bounds"][1],
            n_bins,
        )
        for r_ind, p in enumerate(model_params[inds]):
            if param == "thickness":
                p *= 1000  # unit conversion to m
            # param timeseries
            ax[r_ind, c_ind].hist(p, bins=bins, density=True)
            # true model
            ax[r_ind, c_ind].axvline(true_model[inds][r_ind], c="red")
            # most probable model
            ax[r_ind, c_ind].axvline(probable_model[inds][r_ind], c="purple")
            # bounds
            ax[r_ind, c_ind].axvline(input_ds.attrs[param + "_bounds"][0], c="black")
            ax[r_ind, c_ind].axvline(input_ds.attrs[param + "_bounds"][1], c="black")

            # axis labels
            ax[r_ind, c_ind].set_xlabel(param + " " + str(r_ind + 1))
            # ax[r_ind, c_ind].set_ylabel("count")

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.show()


def resulting_model_histogram(input_ds, results_ds):
    """
    plot the resulting model as velocity vs. depth
    with the histogram of probability for the thickness of the layer
    """
    # use results_ds to get model params
    model_params = results_ds["model_params"].values

    # true model
    true_params = input_ds["model_true"].values

    # define hist bins between bounds
    n_bins = 100
    thickness_bins = (
        np.linspace(
            input_ds.attrs["thickness_bounds"][0],
            input_ds.attrs["thickness_bounds"][1],
            n_bins,
        )
        * 1000
    )  # unit conversion
    vel_s_bins = np.linspace(
        input_ds.attrs["vel_s_bounds"][0], input_ds.attrs["vel_s_bounds"][1], n_bins
    )
    counts = np.zeros((n_bins, n_bins))

    # loop over every resulting model
    # add vel_s 1 to hist bins above thickness
    # add vel_s 2 to hist bins below thickness

    thickness_inds = input_ds["thickness_inds"]
    vel_s_inds = input_ds["vel_s_inds"]

    n_steps = len(results_ds["step"])

    thickness = model_params[thickness_inds] * 1000  # unit conversion to m
    true_thickness = true_params[thickness_inds] * 1000

    thickness_plotting = np.concatenate(
        (
            np.zeros((1, n_steps)),
            thickness,
            np.full((1, n_steps), input_ds.attrs["thickness_bounds"][1]),
        ),
        axis=0,
    )
    true_thickness_plotting = np.concatenate(([0], true_thickness))

    vel_s = model_params[vel_s_inds]
    true_vel_s = true_params[vel_s_inds]

    # for each layer
    # for each sample / step
    total_thickness = np.cumsum(thickness_plotting, axis=0)
    for layer_ind in range(input_ds.attrs["n_layers"] + 1):
        for step_ind in range(n_steps):
            # find bin index closest to layer thickness
            thickness_upper_inds = np.argmin(
                abs(thickness_bins - total_thickness[layer_ind, step_ind])
            )
            thickness_lower_inds = np.argmin(
                abs(thickness_bins - total_thickness[layer_ind + 1, step_ind])
            )
            # find bin index closest to layer vel_s
            vel_s_inds = np.argmin(abs(vel_s_bins - vel_s[layer_ind, step_ind]))

            counts[thickness_upper_inds:thickness_lower_inds, vel_s_inds] += 1

    true_model = []
    true_total_thickness = np.cumsum(true_thickness_plotting)
    true_total_thickness = np.concatenate(
        (true_total_thickness, [input_ds.attrs["thickness_bounds"][1] * 1000])
    )
    for layer_ind in range(input_ds.attrs["n_layers"] + 1):
        true_model.append([true_total_thickness[layer_ind], true_vel_s[layer_ind]])
        true_model.append([true_total_thickness[layer_ind + 1], true_vel_s[layer_ind]])

    fig = plt.figure()
    gs = GridSpec(1, 3, figure=fig)

    # Add subplots with custom spans
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])

    h = ax1.imshow(
        counts,
        norm=LogNorm(),
        extent=[vel_s_bins[0], vel_s_bins[-1], thickness_bins[-1], thickness_bins[0]],
        aspect="auto",
    )

    # plot true model overtop
    true_model = np.array(true_model)
    ax1.plot(true_model[:, 1], true_model[:, 0], c="red")

    fig.colorbar(h, ax=ax1)
    ax1.set_xlabel("vel s (km/s)")
    ax1.set_ylabel("depth (m)")

    # plot thickness histogram
    total_thickness = np.cumsum(thickness, axis=0)
    for ind in range(input_ds.attrs["n_layers"]):
        ax2.hist(
            total_thickness[ind],
            bins=thickness_bins,
            density=True,
            orientation="horizontal",
        )

    ax2.set_ylim(
        [input_ds.attrs["thickness_bounds"][0], input_ds.attrs["thickness_bounds"][1]]
    )
    plt.gca().invert_yaxis()

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.tight_layout()
    plt.show()


def plot_data_pred_histogram(input_ds, results_ds, n_bins=50):
    """ """
    fig, ax = plt.subplots()
    freqs = 1 / input_ds["period"]

    plt.plot(freqs, input_ds["data_true"], zorder=5)

    # yerr = input_ds.attrs["sigma_data"] * input_ds["data_obs"]
    # plt.errorbar(freqs, input_ds["data_true"], yerr, marker="o")
    plt.scatter(freqs, input_ds["data_obs"], zorder=5)

    # estimated error
    yerr = input_ds.attrs["sigma_data"] * results_ds["data_prob"]
    plt.errorbar(freqs, results_ds["data_prob"], yerr, fmt="o", zorder=5)
    # plt.scatter(results_ds["period"], results_ds["data_prob"], zorder=5)

    # flatten data_pred, repeat period
    hist_freqs = np.repeat(freqs, results_ds["data_pred"].shape[1])
    data_preds = results_ds["data_pred"].values.flatten()

    plt.hist2d(hist_freqs, data_preds, bins=n_bins, cmin=1, norm="log")
    # fig.colorbar(im, ax=ax, label="count")

    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("velocity (km/s)")

    plt.legend(["data_true", "data_obs", "data_pred"])
    plt.show()


def compare_results():
    # likelihood of most probable model for diff runs
    # BIC (number of parameters vs. likelihood of best model)
    pass
