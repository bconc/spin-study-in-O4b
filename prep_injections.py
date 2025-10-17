# from https://github.com/tcallister/get-lvk-data/blob/main/prep_injections.py
import numpy as np
import h5py
import json
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15
import sys
import pickle

from priors import chi_effective_prior_from_isotropic_spins
from priors import joint_prior_from_isotropic_spins


# Constants
c = 299792458 # m/s
H_0 = 67900.0 # m/s/MPc
Omega_M = 0.3065 # unitless
Omega_Lambda = 1.0-Omega_M
year = 365.25*24.*3600
target_nSamps = 6000


# Functions
def Hz(z):
    return H_0*np.sqrt(Omega_M*(1.+z)**3.+Omega_Lambda)

def loadInjections(ifar_threshold, snr_threshold):

    all_O4b = 'input/samples-rpo4b_v1-1393286656-29116416.hdf'
    everything_else = 'input/rpo1234-cartesian_spins-semi_o1_o2_o4a-real_o3-T2400110_v3.hdf'

    # Read injection file
    mockDetections = h5py.File(everything_else, 'r')
    mockDetectionsO4b = h5py.File(all_O4b, 'r')

    mockDetections_mid = mockDetections['events'][()]
    mockDetections_dict = {name: mockDetections_mid[name] for name in mockDetections_mid.dtype.names}
    mockDetectionsO4b_mid = mockDetectionsO4b['events'][()]
    mockDetectionsO4b_dict = {name: mockDetectionsO4b_mid[name] for name in mockDetectionsO4b_mid.dtype.names}

    # Total number of trial injections (detected or not)
    nTrials = (mockDetections.attrs['total_generated'] + mockDetectionsO4b.attrs['total_generated'])

    # Read out IFARs and SNRs from search pipelines
    far_1 = mockDetections_dict['o3_gstlal_far']
    far_2 = mockDetections_dict['o3_pycbc_bbh_far']
    far_3 = mockDetections_dict['o3_pycbc_hyperbank_far']
    snr = mockDetections_dict['estimated_optimal_snr_net']
    # For O4b: 'snr_net' is optimal SNR
    snr_O4b = mockDetectionsO4b_dict['snr_net']

    # Determine which events pass IFAR threshold (O3) or SNR threshold (O1/O2)
    # detected_O3 = np.where((far_1<ifar_threshold) | (far_2<ifar_threshold) | (far_3<ifar_threshold))[0]
    # detected_O1O2 = np.where((mockDetections['injections']['name'][()]!=b'o3') & (snr>snr_threshold))[0]
    detected_O1O2O3O4a = np.where(snr>snr_threshold)[0]
    detected_O4b = np.where(snr_O4b>snr_threshold)[0]
    test = np.where(snr>0.0)[0]
    print('SIZE CHECK', detected_O1O2O3O4a.size < test.size,
          len(detected_O4b)/len(snr_O4b), len(detected_O1O2O3O4a)/len(snr))

    # Get properties of detected sources
    mixture_weights_det = np.array(mockDetections_dict['weights'])[detected_O1O2O3O4a]
    m1_det = np.array(mockDetections_dict['mass1_source'])[detected_O1O2O3O4a]
    m2_det = np.array(mockDetections_dict['mass2_source'])[detected_O1O2O3O4a]
    s1x_det = np.array(mockDetections_dict['spin1x'])[detected_O1O2O3O4a]
    s1y_det = np.array(mockDetections_dict['spin1y'])[detected_O1O2O3O4a]
    s1z_det = np.array(mockDetections_dict['spin1z'])[detected_O1O2O3O4a]
    s2x_det = np.array(mockDetections_dict['spin2x'])[detected_O1O2O3O4a]
    s2y_det = np.array(mockDetections_dict['spin2y'])[detected_O1O2O3O4a]
    s2z_det = np.array(mockDetections_dict['spin2z'])[detected_O1O2O3O4a]
    z_det = np.array(mockDetections_dict['redshift'])[detected_O1O2O3O4a]

    # In general, we'll want either dP_draw/(dm1*dm2*dz*da1*da2*dcost1*dcost2) or dP_draw/(dm1*dm2*dz*dchi_eff*dchi_p).
    # In preparation for computing these quantities, divide out by the component draw probabilities dP_draw/(ds1x*ds1y*ds1z*ds2x*ds2y*ds2z)
    # Note that injections are uniform in spin magnitude (up to a_max = 0.998) and isotropic, giving the following:
    dP_ds1x_ds1y_ds1z = (1./(4.*np.pi))*(1./0.998)/(s1x_det**2+s1y_det**2+s1z_det**2)
    dP_ds2x_ds2y_ds2z = (1./(4.*np.pi))*(1./0.998)/(s2x_det**2+s2y_det**2+s2z_det**2)

    # This is dP_draw/(dm1*dm2*dz*ds1x*ds1y*ds1z*ds2x*ds2y*ds2z)
    # precomputed_p_m1m2z_spin = np.array(mockDetections['injections']['sampling_pdf'][()])[detected_full]

    ### Could use this?
    # p_m1_det = jnp.interp(jnp.log10(m1_det), logm_grid, p_m1_grid) / p_m_norm
    # p_m2_det = (1. + bq) * m2_det ** bq / (m1_det ** (1. + bq) - tmp_min ** (1. + bq))
    # p_z_det = dVdz_det * (1. + z_det) ** (kappa - 1.) / p_z_norm
    # R_pop_det = R20 * p_m1_det * p_m2_det * p_z_det * p_Xeff_det
    ###

    precomputed_p_m1m2z_spin = (
        np.exp(mockDetections_dict['lnpdraw_mass1_source_mass2_source_redshift_' +
                                   'spin1x_spin1y_spin1z_spin2x_spin2y_spin2z'][detected_O1O2O3O4a]))
    precomputed_p_m1m2z = precomputed_p_m1m2z_spin/dP_ds1x_ds1y_ds1z/dP_ds2x_ds2y_ds2z

    precomputed_p_m1m2z_O4b = (np.exp(mockDetectionsO4b_dict["lnpdraw_mass1_source"] +
                                         mockDetectionsO4b_dict["lnpdraw_mass2_source_GIVEN_mass1_source"]) *
                               np.exp(mockDetectionsO4b_dict["lnpdraw_z"]))[detected_O4b]

    # Join O1-O4a with O4b
    mixture_weights_det = np.concatenate((mixture_weights_det, mockDetectionsO4b_dict['weights'][detected_O4b]), axis=0)
    precomputed_p_m1m2z = np.concatenate((precomputed_p_m1m2z, precomputed_p_m1m2z_O4b), axis=0)
    m1_det = np.concatenate((m1_det, mockDetectionsO4b_dict['mass1_source'][detected_O4b]), axis=0)
    m2_det = np.concatenate((m2_det, mockDetectionsO4b_dict['mass2_source'][detected_O4b]), axis=0)
    s1x_det = np.concatenate((s1x_det, mockDetectionsO4b_dict['spin1x'][detected_O4b]), axis=0)
    s1y_det = np.concatenate((s1y_det, mockDetectionsO4b_dict['spin1y'][detected_O4b]), axis=0)
    s1z_det = np.concatenate((s1z_det, mockDetectionsO4b_dict['spin1z'][detected_O4b]), axis=0)
    s2x_det = np.concatenate((s2x_det, mockDetectionsO4b_dict['spin2x'][detected_O4b]), axis=0)
    s2y_det = np.concatenate((s2y_det, mockDetectionsO4b_dict['spin2y'][detected_O4b]), axis=0)
    s2z_det = np.concatenate((s2z_det, mockDetectionsO4b_dict['spin2z'][detected_O4b]), axis=0)
    z_det = np.concatenate((z_det, mockDetectionsO4b_dict['z'][detected_O4b]), axis=0)

    return m1_det,m2_det,s1x_det,s1y_det,s1z_det,s2x_det,s2y_det,s2z_det,z_det,precomputed_p_m1m2z,nTrials,mixture_weights_det

def genInjectionFile(ifar_threshold, snr_threshold, filename):

    # Load
    m1_det,m2_det,s1x_det,s1y_det,s1z_det,s2x_det,s2y_det,s2z_det,z_det,p_draw_m1m2z,nTrials,mixture_weights = (
        loadInjections(ifar_threshold,snr_threshold))

    # Derived parameters
    q_det = m2_det/m1_det
    Xeff_det = (m1_det*s1z_det + m2_det*s2z_det)/(m1_det+m2_det)
    Xp_det = np.maximum(np.sqrt(s1x_det**2+s1y_det**2),(3.+4.*q_det)/(4.+3.*q_det)*q_det*np.sqrt(s2x_det**2+s2y_det**2))
    a1_det = np.sqrt(s1x_det**2 + s1y_det**2 + s1z_det**2)
    a2_det = np.sqrt(s2x_det**2 + s2y_det**2 + s2z_det**2)
    cost1_det = s1z_det/a1_det
    cost2_det = s2z_det/a2_det

    # Compute marginal draw probabilities for chi_effective and joint chi_effective vs. chi_p probabilities
    p_draw_xeff = np.zeros(Xeff_det.size)
    p_draw_xeff_xp = np.zeros(Xeff_det.size)

    for i in range(p_draw_xeff.size):
        if i%500==0:
            print(i, '/', p_draw_xeff.size)
        p_draw_xeff[i] = chi_effective_prior_from_isotropic_spins(q_det[i],1.,Xeff_det[i])
        p_draw_xeff_xp[i] = joint_prior_from_isotropic_spins(q_det[i],1.,Xeff_det[i],Xp_det[i],ndraws=10000)

    # Draw probabilities for component spin magnitudes and tilts
    p_draw_a1a2cost1cost2 = (1./2.)**2*(1./0.998)**2*np.ones(a1_det.size)

    # Combine
    pop_reweight = 1./(p_draw_m1m2z*p_draw_xeff_xp)
    pop_reweight_XeffOnly = 1./(p_draw_m1m2z*p_draw_xeff)
    pop_reweight_noSpin = 1./p_draw_m1m2z

    # Also compute factors of dVdz that we will need to reweight these samples during inference later on
    dVdz = 4.*np.pi*Planck15.differential_comoving_volume(z_det).to(u.Gpc**3*u.sr**(-1)).value

    # Store and save
    injectionDict = {
            'm1':m1_det,
            'm2':m2_det,
            'Xeff':Xeff_det,
            'Xp':Xp_det,
            'z':z_det,
            's1z':s1z_det,
            's2z':s2z_det,
            'a1':a1_det,
            'a2':a2_det,
            'cost1':cost1_det,
            'cost2':cost2_det,
            'dVdz':dVdz,
            'p_draw_m1m2z':p_draw_m1m2z,
            'p_m1_m2_z_Xeff': p_draw_m1m2z*p_draw_xeff,
            'p_draw_chiEff':p_draw_xeff,
            'p_draw_chiEff_chiP':p_draw_xeff_xp,
            'p_draw_a1a2cost1cost2':p_draw_a1a2cost1cost2,
            'nTrials':nTrials,
            'mixture_weights': mixture_weights
            }

    with open(filename,'wb') as f:
        pickle.dump(injectionDict,f,protocol=2)

if __name__=="__main__":

    genInjectionFile(1,10.,'input/injectionDict_FAR_1_in_1_O1-O4b.pickle')