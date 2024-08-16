import xarray as xr
import numpy as np


def test_save_data():
    n_steps = 20
    n_params = 5

    ds = xr.Dataset(
        # data_vars={"params": ("step", params_results)},
        coords={"step": np.arange(n_steps), "param": np.arange(n_params)},
        # attrs={}
    )
    ds["params"] = (["step", "param"], np.random.random((n_steps, n_params)))

    print(ds)

    """
    ds = xr.Dataset(
        data_vars={
            "params": ("step", np.arrayself.stored_results["params"]),
            "logL": ("step", self.stored_results["logL"]),
            "rot_mat": ("step", self.stored_results["rot_mat"]),
            "sigma_pd": ("step", self.stored_results["sigma_pd"]),
        },
        coords={"step": np.arange(n_steps - self.n_keep, n_steps)},
        # attrs={}
    )
    """
    assert 1 > n_steps


def test_rotation():
    pass
