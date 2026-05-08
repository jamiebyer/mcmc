# MCMC Bayesian inversion

## Description
- Bayesian inversion of dispersion curves and/or ellipticity spectra.


## Usage
- install the environment with either `conda env create -f environment.yml` or `conda create -n env_name -f requirements.txt`
- (create environment files with `conda env export > environment.yml` or `conda list -e > requirements.txt`)
- **Note**: for disba, python version must be ---


## Algorithm
- optional... optimization inversion to get starting model...

- optional trans-d inversion

- normalization
- proposal distribution (uniform or cauchy...)

- optional stepsize tuning
    - with parameter rotation
        - linear rotation during burn-in
        - rotation using an estimate of the covariance matrix from recorded sampling.
    - using acceptance rate

- perturb parameters randomly selected, one at a time

- forward model using disba
- using Metropolis-Hastings acceptance criteria

- optional parallel tempering

**Data**
- `./src/inversion/data.py`
- generate synthetic data
- periods
- sigma data
    - errors are a percentage of the data

**Model parameterization**
- `./src/inversion/model_params.py`
- Specific model parameterization
- use input vpvs ratio to compute vel_p
- use Garner's relation to compute density
- solving for layer depth and vel_s (layer swapping)

- `./src/inversion/model.py`
- General model information
- perturb params
- get likelihood


**Inversion**
- `./src/inversion/inversion.py`


## Plotting

- `./src/plotting/plot_dispersion_curve.py`