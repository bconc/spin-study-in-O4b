This repository holds code and data release products behind the paper "Gravitational waves reveal the pair-instability mass
gap and constrain nuclear burning in massive stars", https://arxiv.org/pdf/2509.04637. The analysis takes roughly one day to complete on a standard PC, but can be significantly sped up using GPU acceleration when available.

The provided packages are largely based on the code found at this link: https://github.com/tcallister/gwtc3-spin-studies.

## Setting up the environment
To reproduce our results and figures, you can use the provided "environment.yaml" file to create a conda environment with all required packages.

### Step 0. Install conda
If you don’t already have conda installed, you can install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Step 1. Create the environment
$ conda env create -f environment.yaml

### Step 2. Activate the environment
$ conda activate gwtc4-spin-studies

You can deactivate the environment using:
$ conda deactivate

### Step 3. Download the data containing PE posteriors and injections
$ sh download.sh

This command will poulate the directory "input" with the data files used in the population modelling
If you want to rerun any of our analyses, you’ll need to download this input/output data locally

### Step 4. Perform hierarchical Bayesian inference
To infer the hyperparameter posteriors of the model in Eq. 4 in https://arxiv.org/pdf/2509.04637. 
The model is mixutre of a Normal distribution, representing the bulk of the population at primary mass $m<\tilde{m}$, 
and a higher mass spin distribution described via a non-parametric Gaussian process prior: 

$ python run_Xeff_N_GP.py 
this will generate the output "Xeff_N_GP.cdf".

For the model in Eq. 6 of  https://arxiv.org/pdf/2509.04637, i.e., a mixture between a Normal and a uniform type the command:

$ python run_Xeff_N_uniform.py 
this will generate the output "Xeff_N_uniform_independent.cdf".

for the model in Eq. 5, use

$ python run_mixture_gp.py
this will generate the output "Xeff_mixture_O4.cdf".

### Step 5. Analyze posterior distributions
Output files are an ArviZ NetCDF file containing posterior samples, diagnostics, and metadata from the NumPyro MCMC run of the specific model.
They stores the inferred parameter distributions for the population model together with sampler statistics.

The notebook analysis.ipynb provides baseline code to visualize posterior distributions and to compute the differential merger rate as a function of primary masse for the low- and high-spin populations.





