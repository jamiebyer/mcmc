from disba import PhaseSensitivity
import numpy as np
import matplotlib.pyplot as plt
from colour import Color
import matplotlib as mpl


def dispersion_depth_sensitivity():
    # read in best model
    # depth sensitivity for different frequencies

    # how much do the results at a depth change by changing the model a bit?

    # look at true model from inversion

    n_layers = 2

    if n_layers == 1:
        # one layer
        # depth = np.array([0.05])
        # vel_s = np.array([0.4, 1.0])

        depth = np.linspace(0.01, 0.99, 99)
        vel_s = np.array([0.4] * 50 + [1.0] * 50)
    elif n_layers == 2:
        # two layers
        # depth = [0.02, 0.08]
        # depth = np.array([0.05, 0.10])
        # vel_s = np.array([0.2, 0.6, 1.5])

        depth = np.linspace(0.01, 0.99, 99)
        vel_s = np.array([0.2] * 30 + [0.6] * 30 + [1.5] * 40)
    elif n_layers == 3:
        # three layers
        depth = np.array([0.02, 0.04, 0.1])
        vel_s = np.array([0.2, 0.6, 1.0, 1.5])

    # functions needed for forward model
    def get_vel_p(vel_s):
        vel_p = vel_s * 1.75
        return vel_p

    def get_density(vel_p):
        # using Garner's relation
        density = (1741 * np.sign(vel_p) * abs(vel_p) ** (1 / 4)) / 1000
        return density

    """
    get phase dispersion curve for current shear velocities and layer thicknesses.

    :param periods:
    :param velocity model: velocity model for disba has the format
        [thickness (km), vel_p (km/s), vel_s (km/s), density (g/cm3)]
    :param model_params: model params to use to get phase dispersion
    """
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
    # pd = PhaseDispersion(*velocity_model)

    # ps = PhaseSensitivity(*velocity_model.T)
    ps = PhaseSensitivity(thickness, vel_p, vel_s, density)
    # parameters = ["thickness", "velocity_p", "velocity_s", "density"]

    freqs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0]
    # freqs = [0.01, 0.05, 0.1, 0.5, 1.0]

    # freqs = np.logspace(-2, 1.5, 20)

    n_lines = len(freqs)
    cmap = mpl.colormaps["viridis"]

    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, n_lines))

    for ind, f in enumerate(freqs):
        p = 1 / f
        skr = ps(p, mode=0, wave="rayleigh", parameter="thickness")
        plt.plot(skr.kernel, skr.depth, label=str(f), c=colors[ind])

    plt.gca().invert_yaxis()

    plt.xlabel("kernel")
    plt.ylabel("depth (km)")

    plt.legend()

    plt.show()


def example_depth_sensitivity():
    velocity_model = np.array(
        [
            [10.0, 7.00, 3.50, 2.00],
            [10.0, 6.80, 3.40, 2.00],
            [10.0, 7.00, 3.50, 2.00],
            [10.0, 7.60, 3.80, 2.00],
            [10.0, 8.40, 4.20, 2.00],
            [10.0, 9.00, 4.50, 2.00],
            [10.0, 9.40, 4.70, 2.00],
            [10.0, 9.60, 4.80, 2.00],
            [10.0, 9.50, 4.75, 2.00],
        ]
    )

    ps = PhaseSensitivity(*velocity_model.T)
    parameters = ["thickness", "velocity_p", "velocity_s", "density"]
    for p in parameters:
        skr = ps(20.0, mode=0, wave="rayleigh", parameter=p)
        plt.plot(skr.kernel, skr.depth, label=p)

    plt.gca().invert_yaxis()
    plt.legend()

    plt.xlabel("kernel")
    plt.ylabel("depth (km)")

    plt.show()


def hvsr_depth_sensitivity():
    pass


if __name__ == "__main__":
    # dispersion_depth_sensitivity()
    example_depth_sensitivity()
