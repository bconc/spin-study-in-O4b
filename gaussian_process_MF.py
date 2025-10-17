import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax.scipy as jscp 
from jax import random 
from jax import vmap
import numpy as np
from utilities import *
from jax.scipy.special import erf

logm_grid = jnp.linspace(0,2.3,100)

#both mixture and mass distribution as gaussian processes
def Xeff_gaussian_to_uniform_gp2(sampleDict,injectionDict):
    
    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    """
    
    # Sample our hyperparameters
    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation

    logR20 = numpyro.sample("logR20",dist.Uniform(-12,12))
    alpha = numpyro.sample("alpha",dist.Normal(-2,3))
    mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
    mMin = numpyro.sample("mMin",dist.Uniform(5,15))
    bq = numpyro.sample("bq",dist.Normal(0,3))
    kappa = numpyro.sample("kappa",dist.Normal(0,5)) 
    R20 = numpyro.deterministic("R20",10.**logR20)

    logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
    logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
    logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
    logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
    logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))
    logit_mu_eff_lowM = numpyro.sample("logit_mu_eff_lowM",dist.Normal(0,logit_std))
    logit_logsig_eff_lowM = numpyro.sample("logit_logsig_eff_lowM",dist.Normal(0,logit_std))
    logit_max_chi_eff = numpyro.sample("logit_max_chi_eff",dist.Normal(0,logit_std))

    sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,2.,15.)
    log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-3,0.)
    mMax,jac_mMax = get_value_from_logit(logit_mMax,50.,100.)
    log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin,-1,1)
    log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.5,1.5)
    mu_eff_lowM,jac_mu_eff_lowM = get_value_from_logit(logit_mu_eff_lowM,-1,1.)
    logsig_eff_lowM,jac_logsig_eff_lowM = get_value_from_logit(logit_logsig_eff_lowM,-1.5,0.)
    max_chi_eff,jac_max_chi_eff = get_value_from_logit(logit_max_chi_eff,0.1,1.)


    numpyro.deterministic("sig_m1",sig_m1)
    numpyro.deterministic("log_f_peak",log_f_peak)
    numpyro.deterministic("mMax",mMax)
    numpyro.deterministic("log_dmMin",log_dmMin)
    numpyro.deterministic("log_dmMax",log_dmMax)
    numpyro.deterministic("mu_eff_lowM",mu_eff_lowM)
    numpyro.deterministic("logsig_eff_lowM",logsig_eff_lowM)
    numpyro.deterministic("max_chi_eff",max_chi_eff)

    numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))
    numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
    numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))
    numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))
    numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))
    numpyro.factor("p_mu_eff_lowM",logit_mu_eff_lowM**2/(2.*logit_std**2)-jnp.log(jac_mu_eff_lowM))
    numpyro.factor("p_logsig_eff_lowM",logit_logsig_eff_lowM**2/(2.*logit_std**2)-jnp.log(jac_logsig_eff_lowM))
    numpyro.factor("p_max_chi_eff",logit_max_chi_eff**1/(2.*logit_std**2)-jnp.log(jac_max_chi_eff))
#    max_chi_eff=0.47

    # GP
    A=numpyro.sample("A",dist.HalfNormal(5))
    lnB=numpyro.sample("lnB",dist.Normal(-0.5,1))
    B=numpyro.deterministic("B",jnp.exp(lnB))
    
    cov = (A)**2*jnp.exp(-(logm_grid[:,jnp.newaxis]-logm_grid[jnp.newaxis,:])**2/(2.*(B)**2))
    L = jscp.linalg.cholesky(cov+1e-9*jnp.eye(logm_grid.size),lower=True)
    gp_draws = numpyro.sample("gp_draws",dist.Normal(0,1),sample_shape=(logm_grid.size,))
    mixture_fraction_grid = 1./(1.+jnp.exp(-L.dot(gp_draws)))
    numpyro.deterministic("mixture_fraction_grid",mixture_fraction_grid)

    C=numpyro.sample("C",dist.HalfNormal(3))
    lnD=numpyro.sample("lnD",dist.Normal(0,1))
    D=numpyro.deterministic("D",jnp.exp(lnD))
    numpyro.factor("regularise_m1", -(C/jnp.sqrt(D))**2/(2.*(0.61*3)**2))

    covm =(C)**2*jnp.exp(-(logm_grid[:,jnp.newaxis]-logm_grid[jnp.newaxis,:])**2/(2.*(D)**2))
    Lm = jscp.linalg.cholesky(covm+1e-9*jnp.eye(logm_grid.size),lower=True)
    gp_m1_draws = numpyro.sample("gp_m1_draws", dist.Normal(0, 1), sample_shape=(logm_grid.size,)) 
    p_m1_grid= jnp.exp(Lm.dot(gp_m1_draws))  #  ensures positivity
    p_m1_grid  /= jnp.trapezoid(p_m1_grid, logm_grid)  # Normalize
    numpyro.deterministic("p_m1_grid", p_m1_grid)

    # Normalization
    p_m_norm = jnp.interp(jnp.log10(20.), logm_grid, p_m1_grid)
    p_z_norm = (1.+0.2)**kappa
 
    # Read out found injections
    # Note that `pop_reweight` is the inverse of the draw weights for each event
    Xeff_det = injectionDict['Xeff']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_m1_m2_z_Xeff']

    # Compute proposed population weights
    p_Xeff_det_low = truncatedNormal(Xeff_det,mu_eff_lowM,10.**logsig_eff_lowM,-1,1)

    w = 0.1
    p_Xeff_det_high = jnp.ones_like(Xeff_det)
    p_Xeff_det_high = jnp.where(Xeff_det<-max_chi_eff,p_Xeff_det_high*jnp.exp(-(Xeff_det+max_chi_eff)**2/(2.*w**2)),p_Xeff_det_high)
    p_Xeff_det_high = jnp.where(Xeff_det>max_chi_eff,p_Xeff_det_high*jnp.exp(-(Xeff_det-max_chi_eff)**2/(2.*w**2)),p_Xeff_det_high)
    norm = 2*max_chi_eff - jnp.sqrt(2.*np.pi)*w*erf((max_chi_eff-1.)/(jnp.sqrt(2.)*w))
    p_Xeff_det_high /= norm

    mixture_fraction_det = jnp.interp(jnp.log10(m1_det),logm_grid,mixture_fraction_grid)
    p_Xeff_det = p_Xeff_det_high*mixture_fraction_det + p_Xeff_det_low*(1.-mixture_fraction_det)

    p_m1_det = jnp.interp(jnp.log10(m1_det), logm_grid, p_m1_grid)/p_m_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    p_z_det = dVdz_det*(1.+z_det)**(kappa-1.)/p_z_norm 
    R_pop_det = R20*p_m1_det*p_m2_det*p_z_det*p_Xeff_det

    # Form ratio of proposed weights over draw weights
    obs_time_in_years = injectionDict['time']/(365.25*24*3600)
    inj_weights = injectionDict['mixture_weights']*R_pop_det/(p_draw/obs_time_in_years)
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)

    # Penalize
    numpyro.factor("Neff_inj_penalty",jnp.log(1./(1.+(nEff_inj/(4.*nObs))**(-30.))))

    
    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # z_sample: Redshift posterior samples
    # dVdz_sample: Differential comoving volume at each sample location
    # Xeff_sample: Effective spin posterior samples
    # priors: PE priors on each sample
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,Xeff_sample,priors):

        p_Xeff_low = truncatedNormal(Xeff_sample,mu_eff_lowM,10.**logsig_eff_lowM,-1,1)

        w = 0.1
        p_Xeff_high = jnp.ones_like(Xeff_sample)
        p_Xeff_high = jnp.where(Xeff_sample<-max_chi_eff,p_Xeff_high*jnp.exp(-(Xeff_sample+max_chi_eff)**2/(2.*w**2)),p_Xeff_high)
        p_Xeff_high = jnp.where(Xeff_sample>max_chi_eff,p_Xeff_high*jnp.exp(-(Xeff_sample-max_chi_eff)**2/(2.*w**2)),p_Xeff_high)
        norm = 2*max_chi_eff - jnp.sqrt(2.*np.pi)*w*erf((max_chi_eff-1.)/(jnp.sqrt(2.)*w))
        p_Xeff_high /= norm

        mixture_fraction = jnp.interp(jnp.log10(m1_sample),logm_grid,mixture_fraction_grid)
        p_Xeff = p_Xeff_high*mixture_fraction + p_Xeff_low*(1.-mixture_fraction)

        # Compute proposed population weights
        p_m1 =jnp.interp(jnp.log10(m1_sample), logm_grid, p_m1_grid)/p_m_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        p_z = dVdz_sample*(1.+z_sample)**(kappa-1.)/p_z_norm
        R_pop = R20*p_m1*p_m2*p_z*p_Xeff

        mc_weights = R_pop/priors

        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['Xeff'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z_prior']*sampleDict[k]['Xeff_priors'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    min_log_neff=numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Penalize
    numpyro.factor("Neff_penalty",jnp.log(1./(1.+(min_log_neff/0.6)**(-30.))))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))


    
