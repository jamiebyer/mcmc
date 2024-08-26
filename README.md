**MCMC Bayesian inversion**

**Description**
- using parallel tempering

**Usage**
- install the environment with either `conda env create -f environment.yml` or `conda create -n env_name -f requirements.txt`
- (create environment files with `conda env export > environment.yml` or `conda list -e > requirements.txt`)

**Running from terminal**
- how to run from terminal (input variables)
- activate the environment
- To run the inversion call `python run.py`. Results of the inversion are saved to a new folder under the out directory.

    True model variables:
    - poisson_ratio
    - density_params
    - n_data
    - n_layers
    - layer_bounds
    - vel_s_bounds
    - sigma_pd_bounds

    Inversion variables:
    - poisson_ratio
    - density_params
    - n_layers
    - n_chains
    - beta_spacing_factor
    - model_variance
    - n_bins
    - n_burn
    - n_keep
    - n_rot

**Plotting from terminal**
- In terminal call `python plotting.py`






