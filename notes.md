
Things that can be wiggled:

Model setup:
- n_layers
- n_data
- model_depth
- sigma_pd
- poisson_ratio
- density_params


Inversion params:
- bounds on each parameter (p, s velocity, density, layer thickness)


-------

- checking convergence with more than 2 chains?
- uncertainties
- density birch params
- the original generates starting model by adding noise to true params?
- scale starting model noise (pcsd) vs. noise on __


- validate params needs to check diff for layers



- instead of lin_rot, use a PC package?


- rename jacobian functions

- compare lin rot with the equations? what is sigma?

- chain convergence, tempering with beta

- check creating starting model

- normalizing with parameter bounds?

- parallel computing


---------


- how will the uncertainties work
- total depth stay the same?


### OVERVIEW 

- Process below happens for each chain.

- Generate starting model by adding noise to the true model.
- 


- the observed data is a phase dispersion curve (that I make, with added noise)
- forward model takes vs and thickness and returns phase dispersion curve
- for the inversion, we are perturbing the layer thicknesses. get vs from pd






PREM
- PREM500 in comma separated value (CSV) IDV file format
- https://ds.iris.edu/spud/earthmodel/10131390



### EQUATIONS

Likelihood function
$$\mathcal{L}(\textbf{m}) \propto \exp[-\frac{1}{2}\sum_{i=1}^{N_D}N_i\log_{e}|\mathbf{r}_i(\mathbf{m})|^2]$$
- m: model parameters
- N_D: number of data/stations
- r: data residuals
- Paper: Efﬁcient trans-dimensional Bayesian
inversion for geoacoustic proﬁle estimation

$$\frac{V_P}{V_S}=\sqrt{\frac{2-2\sigma}{1 - 2\sigma}}$$