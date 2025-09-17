import numpyro
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import lax
import scipy
import numpy as np

logit_std = 2.5
tmp_max = 100.
tmp_min = 2.

def truncatedNormal(samples,mu,sigma,lowCutoff,highCutoff):

    """
    Jax-enabled truncated normal distribution
    
    Parameters
    ----------
    samples : `jax.numpy.array` or float
        Locations at which to evaluate probability density
    mu : float
        Mean of truncated normal
    sigma : float
        Standard deviation of truncated normal
    lowCutoff : float
        Lower truncation bound
    highCutoff : float
        Upper truncation bound

    Returns
    -------
    ps : jax.numpy.array or float
        Probability density at the locations of `samples`
    """

    a = (lowCutoff-mu)/jnp.sqrt(2*sigma**2)
    b = (highCutoff-mu)/jnp.sqrt(2*sigma**2)
    norm = jnp.sqrt(sigma**2*np.pi/2)*(-erf(a) + erf(b))
    ps = jnp.exp(-(samples-mu)**2/(2.*sigma**2))/norm
    return ps

def smoothlyTruncatedUniform(samples, maxVal, minVal, w):

    """
    Defines a normalized uniform distribution with a smooth exponential truncation
    applied to values above and below a certain threshold.

    Parameters
    ----------
    samples : array
        Values at which to calculate probability
    maxVal : float
        Maximum value
    minVal : float
        Minimum value
    w : float
        Truncation length scale

    Returns
    -------
    ps : array
        Normalized probability density
    """

    # Define unnormalized probability densities
    ps = jnp.ones_like(samples)
    ps = jnp.where(
            samples<minVal,
            jnp.exp(-(samples-minVal)**2/(2*w**2)),
            ps)
    ps = jnp.where(
            samples>maxVal,
            jnp.exp(-(samples-maxVal)**2/(2*w**2)),
            ps)

    # Apply normalization constant
    norm = maxVal - minVal - w*jnp.sqrt(np.pi/2.)*(\
            erf((maxVal-1.)/(jnp.sqrt(2)*w))
            - erf((minVal+1.)/(jnp.sqrt(2)*w)))

    return ps/norm

def massModel(m1,alpha,mu_m1,sig_m1,f_peak,mMax,mMin,dmMax,dmMin):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    p_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Define power-law and peak
    p_m1_pl = (1.+alpha)*m1**(alpha)/(tmp_max**(1.+alpha) - tmp_min**(1.+alpha))
    p_m1_peak = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl)*low_filter*high_filter

def extendedMassModel(m1,alpha1,alpha2,mBreak,mu_m1,sig_m1,f_peak1,mu_m2,sig_m2,f_peak2,mMax,mMin,dmMax,dmMin):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    p_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Broken power law
    #p_bpl_norm = (1.+alpha1)*(1.+alpha2)/(mBreak*(alpha2-alpha1)-tmp_min*jnp.power(tmp_min/mBreak,alpha1)*(1.+alpha2))
    p_bpl_norm = (mBreak**(1.+alpha1)-tmp_min**(1.+alpha1))/mBreak**alpha1/(1.+alpha1) + (tmp_max**(1.+alpha2) - mBreak**(1.+alpha2))/mBreak**alpha2/(1.+alpha2)
    p_m1_broken_pl = jnp.where(m1<mBreak,(m1/mBreak)**alpha1,(m1/mBreak)**alpha2)/p_bpl_norm

    # Peaks
    p_m1_peak1 = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)
    p_m1_peak2 = jnp.exp(-(m1-mu_m2)**2/(2.*sig_m2**2))/jnp.sqrt(2.*np.pi*sig_m2**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    # Apply filters to combined power-law and peak
    return (f_peak1*p_m1_peak1 + f_peak2*p_m1_peak2 + (1.-f_peak1-f_peak2)*p_m1_broken_pl)*low_filter*high_filter

def extendedMassModel_independentMassRatio(m1,m2,alpha1,alpha2,mBreak,mu_m1,sig_m1,f_peak1,mu_m2,sig_m2,f_peak2,mMax,mMin,dmMax,dmMin,beta1,beta2,beta3):

    """
    Baseline primary mass model, described as a mixture between a power law
    and gaussian, with exponential tapering functions at high and low masses

    Parameters
    ----------
    m1 : array or float
        Primary masses at which to evaluate probability densities
    alpha : float
        Power-law index
    mu_m1 : float
        Location of possible Gaussian peak
    sig_m1 : float
        Stanard deviation of possible Gaussian peak
    f_peak : float
        Approximate fraction of events contained within Gaussian peak (not exact due to tapering)
    mMax : float
        Location at which high-mass tapering begins
    mMin : float
        Location at which low-mass tapering begins
    dmMax : float
        Scale width of high-mass tapering function
    dmMin : float
        Scale width of low-mass tapering function

    Returns
    -------
    p_m1s : jax.numpy.array
        Unnormalized array of probability densities
    """

    # Broken power law
    p_bpl_norm = (mBreak**(1.+alpha1)-tmp_min**(1.+alpha1))/mBreak**alpha1/(1.+alpha1) + (tmp_max**(1.+alpha2) - mBreak**(1.+alpha2))/mBreak**alpha2/(1.+alpha2)
    p_m1_broken_pl = jnp.where(m1<mBreak,(m1/mBreak)**alpha1,(m1/mBreak)**alpha2)/p_bpl_norm

    # Peaks
    p_m1_peak1 = jnp.exp(-(m1-mu_m1)**2/(2.*sig_m1**2))/jnp.sqrt(2.*np.pi*sig_m1**2)
    p_m1_peak2 = jnp.exp(-(m1-mu_m2)**2/(2.*sig_m2**2))/jnp.sqrt(2.*np.pi*sig_m2**2)

    # Compute low- and high-mass filters
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMin**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMax**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)

    # Component-wise mass ratio distributions
    p_m2_peak1 = (1.+beta1)*m2**beta1/(m1**(1.+beta1)-tmp_min**(1.+beta1))
    p_m2_peak2 = (1.+beta2)*m2**beta2/(m1**(1.+beta2)-tmp_min**(1.+beta2))
    p_m2_bpl = (1.+beta3)*m2**beta3/(m1**(1.+beta3)-tmp_min**(1.+beta3))

    # Apply filters to combined power-law and peak
    return (f_peak1*p_m1_peak1*p_m2_peak2 \
                + f_peak2*p_m1_peak2*p_m2_peak2 \
                + (1.-f_peak1-f_peak2)*p_m1_broken_pl*p_m2_bpl)\
            *low_filter*high_filter

def broken_pl(z,kappa1,kappa2,zBreak):

    return jnp.where(z<zBreak,((1.+z)/(1.+zBreak))**kappa1,((1.+z)/(1.+zBreak))**kappa2)

def get_value_from_logit(logit_x,x_min,x_max):

    """
    Function to map a variable `logit_x`, defined on `(-inf,+inf)`, to a quantity `x`
    defined on the interval `(x_min,x_max)`.

    Parameters
    ----------
    logit_x : float
        Quantity to inverse-logit transform
    x_min : float
        Lower bound of `x`
    x_max : float
        Upper bound of `x`

    Returns
    -------
    x : float
       The inverse logit transform of `logit_x`
    dlogit_dx : float
       The Jacobian between `logit_x` and `x`; divide by this quantity to convert a uniform prior on `logit_x` to a uniform prior on `x`
    """

    exp_logit = jnp.exp(logit_x)
    x = (exp_logit*x_max + x_min)/(1.+exp_logit)
    dlogit_dx = 1./(x-x_min) + 1./(x_max-x)

    return x,dlogit_dx

def sampleUniformFromLogit(name, minValue, maxValue, logit_std=logit_std):

    """
    Function to sample a uniform bounded prior via an unbounded prior in a transformed
    logit space.

    Parameters
    ----------
    name : `str`
        Name for target parameter
    minValue : `float`
        Minimum allowed value
    maxValue : `float`
        Maximum allowed value
    logit_std : `float`
        Standard deviation for initial sampling prior in logit space.
        This is later undone, but may effect sampling efficiency.
        Default 2.5.

    Returns
    -------
    param : `float`
        Sampled parameter value
    """

    # Draw from unconstrained logit space
    logit_param = numpyro.sample("logit_"+name,
        numpyro.distributions.Normal(0, logit_std))

    # Transform to physical value and get Jacobian
    param, jacobian = get_value_from_logit(logit_param, minValue, maxValue)

    # Undo sampling prior and record result
    numpyro.factor("p_"+name, logit_param**2/(2.*logit_std**2) - jnp.log(jacobian))
    numpyro.deterministic(name, param)
    
    return param


def sampleAutoregressiveProcess(name, reference_grid, reference_index, std_std, ln_tau_mu, ln_tau_std, regularization_std):

    """
    Helper function to sample AR process over a regular grid.

    Parameters
    ----------
    name : `str`
        Name of parameter over which we're building AR process. Will be
        inserted into names of sampling sites
    reference_grid : `jax.numpy.array`
        Regular grid of values over which we're constructing process
    reference_index : `int`
        Index of site from which we will begin sampling left and right
    std_std : `float`
        Standard deviation of prior on AR standard deviation
    ln_tau_mu : `float`
        Mean of prior on log-scale length of AR process
    ln_tau_std : `float`
        Standard deviation of prior on log-scale length
    regularization_std : `float`
        Standard deviation associated with regularization prior on AR hyperparameters 

    Returns
    -------
    fs : `jax.numpy.array`
        AR process evaluated across reference grid
    """

    # First get variance of the process
    ar_std = numpyro.sample("ar_"+name+"_std", numpyro.distributions.HalfNormal(std_std))

    # Next the autocorrelation length
    log_ar_tau = numpyro.sample("log_ar_"+name+"_tau", numpyro.distributions.Normal(ln_tau_mu, ln_tau_std))
    ar_tau = numpyro.deterministic("ar_"+name+"_tau", jnp.exp(log_ar_tau))

    # As discussed in Appendix B, we need a regularizing log-likelihood factor to help stabilize our inference
    numpyro.factor(name+"_regularization", -(ar_std/jnp.sqrt(ar_tau))**2/(2.*regularization_std**2))

    # Sample an initial rate density at reference point
    ln_f_ref_unscaled = numpyro.sample("ln_f_"+name+"_ref_unscaled", numpyro.distributions.Normal(0, 1))
    ln_f_ref = ln_f_ref_unscaled*ar_std

    # Generate forward steps and join to reference value, following the procedure outlined in Appendix A
    # First generate a sequence of unnormalized steps from N(0, 1), then rescale to compute weights and innovations
    deltas = np.diff(reference_grid)[0]
    steps_forward = numpyro.sample(name+"_steps_forward", numpyro.distributions.Normal(0, 1), sample_shape=(reference_grid[reference_index:].size-1, ))
    ws_forward = jnp.sqrt(1.-jnp.exp(-2.*deltas/ar_tau))*ar_std*steps_forward
    phis_forward = jnp.ones(ws_forward.size)*jnp.exp(-deltas/ar_tau)
    final, ln_f_high = lax.scan(build_ar1, ln_f_ref, jnp.transpose(jnp.array([phis_forward, ws_forward])))
    ln_fs = jnp.append(ln_f_ref, ln_f_high)

    # Generate backward steps and prepend to forward steps above following an analogous procedure
    steps_backward = numpyro.sample(name+"_steps_backward", numpyro.distributions.Normal(0, 1), sample_shape=(reference_grid[:reference_index].size, ))
    ws_backward = jnp.sqrt(1.-jnp.exp(-2.*deltas/ar_tau))*ar_std*steps_backward
    phis_backward = jnp.ones(ws_backward.size)*jnp.exp(-deltas/ar_tau)
    final, ln_f_low = lax.scan(build_ar1, ln_f_ref, jnp.transpose(jnp.array([phis_backward, ws_backward])))
    ln_fs = jnp.append(ln_f_low[::-1], ln_fs)

    fs = jnp.exp(ln_fs)
    numpyro.deterministic("fs_"+name,fs)

    return fs


def build_ar1(total, new_element):

    """
    Helper function to iteratively construct an AR process, given a previous value and a new parameter/innovation pair. Used together with `jax.lax.scan`

    Parameters
    ----------
    total : float
        Processes' value at the previous iteration
    new_element : tuple
        Tuple `(c, w)` containing new parameter/innovation; see Eq. 4 of the associated paper

    Returns
    -------
    total : float
        AR process value at new point
    """

    c, w = new_element
    total = c*total+w
    return total, total


def compute_prior_params(dR_max, dR_event, deltaX, N_events):

    """
    Function to compute quantities appearing in our prior on AR(1) process variances and autocorrelation lengths,
    following discussion in Appendix B

    Parameters
    ----------
    dR_max : float
        Estimate of the maximum allowed variation in the merger rate across the domain
    dR_event : float
        Estimate of the maximum allowed variation in the merger rate between event locations
    deltaX : float
        Domain width
    N_events : int
        Number of observations in our sample

    Returns
    -------
    Sigma_sig : float
        Standard deviation to be used in a Gaussian prior on AR(1) process standard deviation `sigma`
    Mu_ln_tau : float
        Mean to be used in a Gaussian prior on AR(1) process' log-autocorrelation length
    Sig_ln_tau : float
        Standard deviation to be used in a Gaussian prior on AR(1) process' log-autocorrelation length
    Sigma_ratio : float
        Standard deviation to be used in a Gaussian regularization prior on the ratio `sigma/sqrt(tau)`
    """

    # Compute the 99th percentile of a chi-squared distribution
    q_99 = scipy.special.gammaincinv(1/2, 0.99)

    # Compute standard deviation on `sigma` prior, see Eq. B21
    Sigma_sig = np.log(dR_max)/(2.*q_99**0.5*scipy.special.erfinv(0.99))

    # Expected minimum spacing between events; see Eq. B29
    dx_min = -(deltaX/N_events)*np.log(1.-(1.-np.exp(-N_events))/N_events)

    # Mean and standard deviation on `ln_tau` prior, see Eqs. B26 and B30
    Mu_ln_tau = np.log(deltaX/2.)
    Sigma_ln_tau = (np.log(dx_min) - Mu_ln_tau)/(2**0.5*scipy.special.erfinv(1.-2*0.99))

    # Standard deviation on ratio, see Eq. B25
    Sigma_ratio = (np.log(dR_event)/(2.*scipy.special.erfinv(0.99)))*np.sqrt(N_events/(q_99*deltaX))

    return Sigma_sig, Mu_ln_tau, Sigma_ln_tau, Sigma_ratio

if __name__=="__main__":

    dR_max = 100
    dR_event = 1.5
    N = 35
    Delta_Xeff = 2.
    Xeff_std_std, Xeff_ln_tau_mu, Xeff_ln_tau_std, Xeff_regularization_std = compute_prior_params(dR_max, dR_event, Delta_Xeff, N)

    print(Xeff_std_std, Xeff_ln_tau_mu, Xeff_ln_tau_std)
