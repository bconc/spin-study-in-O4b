#!/usr/bin/env python3

import numpyro
import jax
nChains = 10
numpyro.set_host_device_count(nChains)
numpyro.set_platform(platform='gpu')
from numpyro.infer import NUTS, MCMC
#from jax.config import config
jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
import arviz as az
from gaussian_process_O4 import Xeff_and_m1_gp_HN_prior
from getData_O4 import *
import h5py
import sys

# Get dictionaries holding injections and posterior samples
#injectionDict = getInjections(reweight=False)
#sampleDict = getSamples(sample_limit=4000,reweight=False)

injectionDict = getInjections(O4=True)
sampleDict = getSamples(sample_limit=3000,O4=True)

# Set up NUTS sampler over our likelihood
kernel = NUTS(Xeff_and_m1_gp_HN_prior)
mcmc = MCMC(kernel,num_warmup=500,num_samples=1500,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(119)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"Xeff_N_GP_nCh10.cdf")

