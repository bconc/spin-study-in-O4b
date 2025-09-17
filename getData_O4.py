import numpy as np
from astropy.cosmology import Planck15
import astropy.units as u
import os,sys
dirname = os.path.dirname(__file__)

def getInjections(O4=False, sample_limit=False, min_mass=3, max_mass=None):

    """
    Function to load and preprocess found injections for use in numpyro likelihood functions.

    Parameters
    ----------
    O4 : bool
        Declare whether to include O4 data

    Returns
    ------- 
    injectionDict : dict
        Dictionary containing found injections and associated draw probabilities, for downstream use in hierarchical inference
    """

    # Load injections
    if O4:
        injectionFile = os.path.join(dirname,"./input/injectionDict_O1O2O3O4a_FAR_1_in_1_semianalytic_SNR_10.pickle")
        injectionDict = np.load(injectionFile,allow_pickle=True)
    else:
        injectionFile = os.path.join(dirname,"./input/injectionDict_O1O2O3_FAR_1_in_1_semianalytic_SNR_10.pickle")
        injectionDict = np.load(injectionFile,allow_pickle=True)

    # Convert all lists to numpy arrays
    for key in injectionDict:
        if key!='nTrials' and key!='time' and key!='checksum':
            injectionDict[key] = np.array(injectionDict[key])

    # Select desired mass range
    to_keep = injectionDict['m2']>min_mass
    if max_mass:
        to_keep *= injectionDict['m1']<max_mass
    for key in injectionDict:
        if key!='nTrials' and key!='time' and key!='checksum':
            injectionDict[key] = injectionDict[key][to_keep]

    if sample_limit:
        originalCount = injectionDict['m1'].size
        injections_to_keep = np.random.choice(np.arange(injectionDict['m1'].size), size=sample_limit)
        for key in injectionDict:
            if key!='nTrials' and key!='time' and key!='checksum':
                injectionDict[key] = injectionDict[key][injections_to_keep]

        injectionDict['nTrials'] *= sample_limit/originalCount

    return injectionDict

def getSamples(sample_limit=2000, bbh_only=True, O4=False, min_mass=3, max_mass=None):

    """
    Function to load and preprocess BBH posterior samples for use in numpyro likelihood functions.
    
    Parameters
    ----------
    sample_limit : int
        Number of posterior samples to retain for each event, for use in population inference (default 2000)
    bbh_only : bool
        If True, will exclude samples for BNS, NSBH, and mass-gap events (default True)
    O4 : bool
        Declare whether to include O4 data

    Returns
    -------
    sampleDict : dict
        Dictionary containing posterior samples, for downstream use in hierarchical inference
    """

    # Load dictionary with preprocessed posterior samples
    sampleFile = os.path.join(dirname,"./input/sampleDict_FAR_1_in_1_yr.pickle")
    sampleDict = np.load(sampleFile,allow_pickle=True)

    # Remove non-BBH events, if desired
    non_bbh = ['GW170817','S190425z','S190426c','S190814bv','S190917u','S200105ae','S200115j']
    if bbh_only:
        for event in non_bbh:
            print("Removing ",event)
            sampleDict.pop(event)

    if O4:
        print('O4 included')
        O4_sampleFile = os.path.join(dirname,"./input/sampleDict_O4a_cat_4c4fd2cef_717_date_250401.npy")
        O4_sampleDict = np.load(O4_sampleFile,allow_pickle=True)[()]

        out_of_sample = ['GW230529_181500']#, 'GW231123_135430']
        for event in out_of_sample:
            print('Removing',event)
            if event in O4_sampleDict.keys():
                print("Popping: ",event)
                O4_sampleDict.pop(event)

        for k,v in O4_sampleDict.items():
            sampleDict[k] = v

    # Loop across events
    for event in sampleDict:

        # Uniform draw weights, trimming mass range as necessary
        draw_weights = np.ones(sampleDict[event]['m1'].size)/sampleDict[event]['m1'].size
        draw_weights[sampleDict[event]['m2']<min_mass] = 0
        if max_mass:
            draw_weights[sampleDict[event]['m1']>max_mass] = 0
        sampleDict[event]['downselection_Neff'] = np.sum(draw_weights)**2/np.sum(draw_weights**2)

        # Randomly downselect to the desired number of samples       
        inds_to_keep = np.random.choice(np.arange(sampleDict[event]['m1'].size),size=sample_limit,replace=True,p=draw_weights/np.sum(draw_weights))
        for key in sampleDict[event].keys():
            if key not in ['downselection_Neff', 'log_prior', 'samples']:
                sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

    return sampleDict


def get_stochastic_data(O4=True):

    """
    Helper function used by `run_birefringence_variable_evolution.py` to load data.

    Parameters
    ----------
    trim_nans : bool
        If True, will remove frequencies that have been notched by data quality flags (default True)

    Returns
    -------
    spectra_dict : dict
        Dictionary containing frequencies, cross-correlation measurements, and uncertainty spectra for all baselines and observing runs
    """

    H1L1_O1_freqs, H1L1_O1_Ys, H1L1_O1_sigmas = np.loadtxt('./../input/H1L1_O1.dat', skiprows=1, unpack=True)
    H1L1_O2_freqs, H1L1_O2_Ys, H1L1_O2_sigmas = np.loadtxt('./../input/H1L1_O2.dat', skiprows=1, unpack=True)
    H1L1_O3_freqs, H1L1_O3_Ys, H1L1_O3_sigmas = np.loadtxt('./../input/H1L1_O3.dat', skiprows=1, unpack=True)
    H1V1_O3_freqs, H1V1_O3_Ys, H1V1_O3_sigmas = np.loadtxt('./../input/H1V1_O3.dat', skiprows=1, unpack=True)
    L1V1_O3_freqs, L1V1_O3_Ys, L1V1_O3_sigmas = np.loadtxt('./../input/L1V1_O3.dat', skiprows=1, unpack=True)

    spectra_dict = {\
        'H1L1_O1':[H1L1_O1_freqs, H1L1_O1_Ys, H1L1_O1_sigmas],
        'H1L1_O2':[H1L1_O2_freqs, H1L1_O2_Ys, H1L1_O2_sigmas],
        'H1L1_O3':[H1L1_O3_freqs, H1L1_O3_Ys, H1L1_O3_sigmas],
        'H1V1_O3':[H1V1_O3_freqs, H1V1_O3_Ys, H1V1_O3_sigmas],
        'L1V1_O3':[L1V1_O3_freqs, L1V1_O3_Ys, L1V1_O3_sigmas],
        }

    if O4:

        H1L1_O4a_data = np.load('./../input/point_estimate_sigma_spectra_alpha_0.0_fref_25_1368979913-20476105.npz')
        H1L1_O4a_freqs = H1L1_O4a_data['frequencies']
        H1L1_O4a_Ys = np.real(H1L1_O4a_data['point_estimate_spectrum'])
        H1L1_O4a_sigmas = H1L1_O4a_data['sigma_spectrum']

        spectra_dict['H1L1_O4a'] = [H1L1_O4a_freqs, H1L1_O4a_Ys, H1L1_O4a_sigmas]

    return spectra_dict


if __name__=="__main__":

    """
    #test = getInjections(O4=False)
    #print(test['time']/(365.25*24*3600))

    min_max_m2_sample = 100
    for key in test.keys():
        m2s = test[key]['m2']
        new_max = np.max(m2s)
        if new_max < min_max_m2_sample:
            min_max_m2_sample = new_max
    print(min_max_m2_sample)
    """

    """
    test = getInjections(sample_limit=2000000, O4=True)
    weights = test['mixture_weights']
    print(np.quantile(weights, 0.1), np.quantile(weights,0.9))
    """

    #test = getInjections(O4=True)#, sample_limit=1000)
    #print(test.keys())

    test = getSamples(O4=True)
    print(len(test))

    """
    test = getSamples(sample_limit=10000, O4=True)
    n_samples_below_max = np.inf
    for event in test:

        m1 = test[event]['m1']
        n_samples = m1[m1<200].size
        if n_samples<n_samples_below_max:
            n_samples_below_max = n_samples

    print(n_samples_below_max)
    """

