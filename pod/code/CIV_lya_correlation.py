#### function for CIV and lya cross-correlation
#### I move all of my modified functions here in order to better organize them

import sys

'''
xi_sum_CIV_lya
compute_xi_all_CIV_lya
compute_xi_all_CIV_CIV
compute_xi_CIV_lya
compute_xi_CIV_lya_double_bin
create_lya_forest
create_metal_forest_tau
calc_igm_Zeff
create_lya_forest_short
create_metal_forest_short
imap_unordered_bar
interp_likelihood_covar_nproc
likelihood_calc
fv_logZ_eff_grid
plot_mcmc_fv_logZ_eff
'''

sys.path.insert(0, "/home/xinsheng/enigma/CIV_forest/") # path of enigma
sys.path.insert(0, "/home/xinsheng/enigma/enigma_git/enigma/reion_forest/")# path of CIV_forest

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table
import glob
import os
from scipy import special
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage import zoom
import mpmath
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15
from astropy import constants as const
from astropy import units as u
from tqdm.auto import tqdm
from sklearn.neighbors import KDTree
from linetools.lists.linelist import LineList
from linetools.abund import solar as labsol
from IPython import embed
from astropy.io import fits
from astropy.table import hstack, vstack
from compute_model_grid_civ import read_model_grid
import halos_skewers
from utils import *
from metal_corrfunc import *
from multiprocessing import Pool
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, RegularGridInterpolator
from scipy import optimize
import emcee
import corner
from matplotlib.ticker import AutoMinorLocator
import inference
import time

def xi_sum_CIV_lya(ind, dist, delta_f_CIV, delta_f_lya , gpm,v_lo, v_hi, nskew, npix_forest):

    npix_sum = np.zeros(nskew, dtype=int)
    flux_sum = np.zeros(nskew)
    for idx in range(npix_forest):
        ibin = (dist[idx] > v_lo) & (dist[idx] <= v_hi)
        n_neigh = np.sum(ibin)
        ind_neigh = (ind[idx])[ibin]
        tmp = np.tile(delta_f_CIV[:, idx]*gpm[:,idx], (n_neigh, 1)).T * (delta_f_lya[:, ind_neigh]*gpm[:, ind_neigh])
        ntmp = np.tile(gpm[:,idx], (n_neigh, 1)).T * gpm[:, ind_neigh]
        flux_sum += np.sum(tmp, axis=1)
        npix_sum += np.sum(ntmp, axis=1)

    xi = (npix_sum > 0)*(flux_sum/(npix_sum + (npix_sum == 0))) # xi = sum of delta flux / sum of pixels

    return xi, npix_sum

def compute_xi_all_CIV_lya(params_CIV, skewers_CIV, params_lya, skewers_lya, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr=None, sampling=None, \
                   cgm_dict=None, metal_dndz_func=None, cgm_seed=None, want_hires=True, type = 'normal'):

    vel_lores_CIV, (flux_lores_tot_CIV, flux_lores_igm_CIV, flux_lores_cgm_CIV), \
    vel_hires_CIV, (flux_hires_tot_CIV, flux_hires_igm_CIV, flux_hires_cgm_CIV), \
    (oden_CIV, v_los_CIV, T_CIV, x_metal_CIV), cgm_tup_CIV, tau_CIV = create_metal_forest_tau(params_CIV, skewers_CIV, logZ, fwhm, metal_ion, sampling=sampling, \
                                                                         cgm_dict=cgm_dict, metal_dndz_func=metal_dndz_func, seed=cgm_seed, type = type)

    vel_lores_lya, (flux_lores_tot_lya, flux_lores_igm_lya, flux_lores_cgm_lya), \
    vel_hires_lya, (flux_hires_tot_lya, flux_hires_igm_lya, flux_hires_cgm_lya), \
    (oden_lya, T_lya, x_metal_lya), cgm_tup_lya, tau_igm_lya = create_lya_forest(params_lya, skewers_lya, fwhm, sampling=sampling)

    mean_flux_tot_CIV = np.mean(flux_hires_tot_CIV)
    delta_f_tot_CIV = (flux_hires_tot_CIV - mean_flux_tot_CIV)/mean_flux_tot_CIV
    print('mean flux of CIV:', mean_flux_tot_CIV)
    print('mean delta_flux of CIV:', np.mean(delta_f_tot_CIV))

    mean_flux_tot_lya = np.mean(flux_hires_tot_lya)
    delta_f_tot_lya = (flux_hires_tot_lya - mean_flux_tot_lya)/mean_flux_tot_lya
    print('mean flux of lya:', mean_flux_tot_lya)
    print('mean delta_flux of lya:', np.mean(delta_f_tot_lya))

    # xi_tot is an array of 2PCF of each skewer
    (vel_mid, xi_tot, npix_tot, xi_zero_lag_tot) = compute_xi_CIV_lya(delta_f_tot_CIV, delta_f_tot_lya, vel_hires_CIV, vel_hires_lya,vmin_corr, vmax_corr, dv_corr)
    xi_mean_tot = np.mean(xi_tot, axis=0) # 2PCF averaged from all the skewers, i.e the final quoted 2PCF

    return vel_mid, xi_mean_tot, xi_tot, npix_tot


def compute_xi_all_CIV_CIV(params_1, skewers_1, params_2, skewers_2, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr = None, sampling=None, \
                   cgm_dict=None, metal_dndz_func=None, cgm_seed=None, want_hires=True, type='normal'):

#### compute the auto_correlation

    vel_lores_1, (flux_lores_tot_1, flux_lores_igm_1, flux_lores_cgm_1), \
    vel_hires_1, (flux_hires_tot_1, flux_hires_igm_1, flux_hires_cgm_1), \
    (oden_1, v_los_1, T_1, x_metal_1), cgm_tup_1, tau_1 = create_metal_forest_tau(params_1, skewers_1, logZ, fwhm, metal_ion, sampling=sampling, \
                                                                          cgm_dict=cgm_dict, metal_dndz_func=metal_dndz_func, seed=cgm_seed,type = type)

    vel_lores_2, (flux_lores_tot_2, flux_lores_igm_2, flux_lores_cgm_2), \
    vel_hires_2, (flux_hires_tot_2, flux_hires_igm_2, flux_hires_cgm_2), \
    (oden_2, v_los_2, T_2, x_metal_2), cgm_tup_2, tau_2 = create_metal_forest_tau(params_2, skewers_2, logZ, fwhm, metal_ion, sampling=sampling, \
                                                                          cgm_dict=cgm_dict, metal_dndz_func=metal_dndz_func, seed=cgm_seed, type = type)

    # Compute mean flux and delta_flux
    mean_flux_tot_1 = np.mean(flux_hires_tot_1)
    delta_f_tot_1 = (flux_hires_tot_1 - mean_flux_tot_1)/mean_flux_tot_1
    print('mean flux of CIV:', mean_flux_tot_1)
    print('mean delta_flux of CIV:', np.mean(delta_f_tot_1))

    mean_flux_tot_2 = np.mean(flux_hires_tot_2)
    delta_f_tot_2 = (flux_hires_tot_2 - mean_flux_tot_2)/mean_flux_tot_2
    print('mean flux of CIV:', mean_flux_tot_2)
    print('mean delta_flux of CIV:', np.mean(delta_f_tot_2))

    # xi_tot is an array of 2PCF of each skewer
    (vel_mid, xi_tot, npix_tot, xi_zero_lag_tot) = compute_xi_CIV_lya(delta_f_tot_1, delta_f_tot_2, vel_hires_1, vel_hires_2, vmin_corr, vmax_corr, dv_corr)
    xi_mean_tot = np.mean(xi_tot, axis=0) # 2PCF averaged from all the skewers, i.e the final quoted 2PCF

    return vel_mid, xi_mean_tot, xi_tot, npix_tot


def compute_xi_CIV_lya(delta_f_in_CIV, delta_f_in_lya, vel_spec_CIV, vel_spec_lya, vmin, vmax, dv, gpm=None, progress=False):

    """
    Args:
        delta_f_in_CIV, delta_f_in_lya (float ndarray), shape (nskew, nspec) or (nspec,):
            Flux contrast array for CIV and lya
        vel_spec_CIV and vel_spec_lya (float ndarray): shape (nspec,)
            Velocities for flux contrast for CIV and lya
        vmin (float):
            Minimum velocity for correlation function velocity grid. This should be a positive number that shold not
            be set to zero, since we deal with the zero lag velocity correlation function separately.
        vmax (float):
            Maximum velocity for correlation function velocity grid. Must be a positive number.
        dv (float):
            Velocity binsize for corrleation functino velocity grid
        gpm (boolean ndarray), same shape as delta_f, Optional
            Good pixel mask (True= Good) for the delta_f_in array. Bad pixels will not be used for correlation function
            computation.
        progress (bool): Optional
            If True then show a progress bar

    Returns:
        v_mid, xi, npix, xi_zero_lag

        v_mid (float ndarray): shape = (ncorr,)
             Midpoint of the bins in the velocity grid for which the correlation function is evaluated. Here
             ncorr = (int(round((vmax - vmin)/dv) + 1)
        xi (float ndarray): shape = (nskew, ncorr)
             Correlation function of each of the nskew input spectra
        npix (float ndarray): shape = (ncorr,)
             Number of spectra pixels contributing to the correlation function estimate in each of the ncorr
             correlation function velocity bins
        xi_zero_lag (float ndarray): shape = (nskew,)
             The zero lage correlation function of each input skewer.

    """

    # This deals with the case where the input delta_f is a single spectrum
    if(len(delta_f_in_CIV.shape)==1):
        delta_f_CIV = delta_f_in_CIV.reshape(1,delta_f_in_CIV.size)
        gpm_use_CIV = np.ones_like(delta_f_CIV,dtype=bool)
    else:
        delta_f_CIV = delta_f_in_CIV
        gpm_use_CIV =  np.ones_like(delta_f_CIV,dtype=bool)
    nskew_CIV, nspec_CIV = delta_f_CIV.shape

    # This deals with the case where the input delta_f is a single spectrum
    if(len(delta_f_in_lya.shape)==1):
        delta_f_lya = delta_f_in_lya.reshape(1,delta_f_in_lya.size)
        gpm_use_lya = np.ones_like(delta_f_lya,dtype=bool)
    else:
        delta_f_lya = delta_f_in_lya
        gpm_use_lya =  np.ones_like(delta_f_lya,dtype=bool)
    nskew_lya, nspec_lya = delta_f_lya.shape

    # Correlation function velocity grid, using the mid point values
    ngrid = int(round((vmax - vmin)/dv) + 1) # number of grid points including vmin and vmax
    v_corr = vmin + dv * np.arange(ngrid)
    v_lo = v_corr[:-1] # excluding the last point (=vmax)
    v_hi = v_corr[1:] # excluding the first point (=vmin)
    v_mid = (v_hi + v_lo)/2.0
    ncorr = v_mid.size

    # This computes all pairs of distances
    data = np.array([vel_spec_CIV])
    data = data.transpose()
    tree = KDTree(data)
    npix_forest = len(vel_spec_CIV)

    xi = np.zeros((nskew_CIV, ncorr)) # storing the CF of each skewer, rather than the CF of all skewers
    npix = np.zeros((nskew_CIV, ncorr), dtype=int) # number of pixels that contribute to the CF

    # looping through each velocity bin and computing the 2PCF
    for iv in range(ncorr):
        # Grab the list of pixel neighbors within this separation
        ind, dist = tree.query_radius(data, v_hi[iv], return_distance=True)
        xi[:, iv], npix[:, iv] = xi_sum_CIV_lya(ind, dist, delta_f_CIV, delta_f_lya, gpm_use_CIV, v_lo[iv], v_hi[iv], nskew_CIV, npix_forest)

    ngood = np.sum(gpm_use_CIV, axis=1)
    xi_zero_lag = (ngood > 0)*np.sum(delta_f_lya*delta_f_CIV*gpm_use_CIV, axis=1)/(ngood + (ngood == 0.0))
    return (v_mid, xi, npix, xi_zero_lag)

def compute_xi_CIV_lya_double_bin(delta_f_in_CIV, delta_f_in_lya, vel_spec_CIV, vel_spec_lya, vmin, vmax, dv1, dv2, v_end, gpm=None, progress=False):

    """
    Args:
        delta_f_in_CIV, delta_f_in_lya (float ndarray), shape (nskew, nspec) or (nspec,):
            Flux contrast array for CIV and lya
        vel_spec_CIV and vel_spec_lya (float ndarray): shape (nspec,)
            Velocities for flux contrast for CIV and lya
        vmin (float):
            Minimum velocity for correlation function velocity grid. This should be a positive number that shold not
            be set to zero, since we deal with the zero lag velocity correlation function separately.
        vmax (float):
            Maximum velocity for correlation function velocity grid. Must be a positive number.
        dv (float):
            Velocity binsize for corrleation functino velocity grid
        gpm (boolean ndarray), same shape as delta_f, Optional
            Good pixel mask (True= Good) for the delta_f_in array. Bad pixels will not be used for correlation function
            computation.
        progress (bool): Optional
            If True then show a progress bar

    Returns:
        v_mid, xi, npix, xi_zero_lag

        v_mid (float ndarray): shape = (ncorr,)
             Midpoint of the bins in the velocity grid for which the correlation function is evaluated. Here
             ncorr = (int(round((vmax - vmin)/dv) + 1)
        xi (float ndarray): shape = (nskew, ncorr)
             Correlation function of each of the nskew input spectra
        npix (float ndarray): shape = (ncorr,)
             Number of spectra pixels contributing to the correlation function estimate in each of the ncorr
             correlation function velocity bins
        xi_zero_lag (float ndarray): shape = (nskew,)
             The zero lage correlation function of each input skewer.

    """
    # This deals with the case where the input delta_f is a single spectrum
    if(len(delta_f_in_CIV.shape)==1):
        delta_f_CIV = delta_f_in_CIV.reshape(1,delta_f_in_CIV.size)
        gpm_use_CIV = np.ones_like(delta_f_CIV,dtype=bool)
    else:
        delta_f_CIV = delta_f_in_CIV
        gpm_use_CIV =  np.ones_like(delta_f_CIV,dtype=bool)
    nskew_CIV, nspec_CIV = delta_f_CIV.shape

    # This deals with the case where the input delta_f is a single spectrum
    if(len(delta_f_in_lya.shape)==1):
        delta_f_lya = delta_f_in_lya.reshape(1,delta_f_in_lya.size)
        gpm_use_lya = np.ones_like(delta_f_lya,dtype=bool)
    else:
        delta_f_lya = delta_f_in_lya
        gpm_use_lya =  np.ones_like(delta_f_lya,dtype=bool)
    nskew_lya, nspec_lya = delta_f_lya.shape

    # Correlation function velocity grid, using the mid point values
    ngrid1 = int(round((v_end - vmin)/dv1) + 1) # number of grid points including vmin and vmax
    ngrid2 = int(round((vmax - v_end)/dv2))
    v_corr1 = vmin + dv1 * np.arange(ngrid1)
    v_corr2 = v_end + dv2 + dv2 * np.arange(ngrid2)
    v_corr = np.concatenate((v_corr1, v_corr2))
    v_lo = v_corr[:-1] # excluding the last point (=vmax)
    v_hi = v_corr[1:] # excluding the first point (=vmin)
    v_mid = (v_hi + v_lo)/2.0
    ncorr = v_mid.size

    # This computes all pairs of distances
    data = np.array([vel_spec_CIV])
    data = data.transpose()
    tree = KDTree(data)
    npix_forest = len(vel_spec_CIV)

    xi = np.zeros((nskew_CIV, ncorr)) # storing the CF of each skewer, rather than the CF of all skewers
    npix = np.zeros((nskew_CIV, ncorr), dtype=int) # number of pixels that contribute to the CF

    # looping through each velocity bin and computing the 2PCF
    for iv in range(ncorr):
        # Grab the list of pixel neighbors within this separation
        ind, dist = tree.query_radius(data, v_hi[iv], return_distance=True)
        xi[:, iv], npix[:, iv] = xi_sum_CIV_lya(ind, dist, delta_f_CIV, delta_f_lya, gpm_use_CIV, v_lo[iv], v_hi[iv], nskew_CIV, npix_forest)

    ngood = np.sum(gpm_use_CIV, axis=1)
    xi_zero_lag = (ngood > 0)*np.sum(delta_f_lya*delta_f_CIV*gpm_use_CIV, axis=1)/(ngood + (ngood == 0.0))
    return (v_mid, xi, npix, xi_zero_lag)


def create_lya_forest(params, skewers, logZ, fwhm, metal_ion='C IV', z=None, sampling=3.0, cgm_dict=None, seed=None):
    """
        Generate lya line forest.
        Args:
            params (astropy table)
            skewers (astropy table)
            fwhm (float): in km/s
            z (float): redshift
            sampling (float): number of pixels per resolution element

            ::For incorporating CGM absorbers::
            cgm_dict (dictionary): containing parameters for the distribution of CGM absorbers.
            metal_dndz_func (function): used to compute dndz of CGM absorber and for drawing random W values. See create_metal_cgm().
            seed (int): random seed for drawing CGM absorbers.
    """

    # ~0.00014 sec to generate one skewer
    if z is None:
        z = params['z'][0]

    dvpix = fwhm/sampling

    # Size of skewers and pixel scale at sim resolution
    vside = params['VSIDE'][0] # box size in km/s
    Ng = params['Ng'][0]       # number of grids on each side of the box
    dvpix_hires = vside/Ng     # pixel scale of the box in km/s
    nskew =len(skewers)

    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    tau0, f_ratio, v_metal, nh_bar = metal_tau0(metal_ion, z, logZ, cosmo=Planck15, X=0.76)
    #tau0, nh_bar = lya_tau0(z, cosmo=Planck15, X=0.76)
    # note that tau0 is obtained from the stronger blue line

    # Pad the skewer for the convolution
    npad = int(np.ceil((7.0*fwhm + v_metal.value)/dvpix_hires)) ## v_metal = 0 and f_ratio = 1 is possible?
    v_pad = npad*dvpix_hires
    pad_tuple = ((0,0), (npad, npad))
    #tau_igm = np.pad(tau0*skewers['TAU'].data, pad_tuple, 'wrap')
    tau_igm = np.pad(skewers['TAU'].data, pad_tuple, 'wrap')

    oden_pad = np.pad(skewers['ODEN'].data, pad_tuple, 'wrap')
    T_pad = np.pad(skewers['T'].data, pad_tuple, 'wrap')
    v_los_pad = np.pad(skewers['VEL_Z'].data, pad_tuple, 'wrap')

    # Determine the velocity coverage including padding, etc.
    v_min = 0.0
    v_max = vside
    vel_pad = (v_min - v_pad) + np.arange(Ng + 2*npad)*dvpix_hires
    # For the high-resolution spectrum take the valid region and recenter about zero
    iobs_hires = (vel_pad >= v_min) & (vel_pad <= v_max)
    vel_hires = vel_pad[iobs_hires]
    nhires = vel_hires.size
    # For the observed spectrum take the valid region and recenter about zero
    vel_obs_pad = np.arange(vel_pad.min(),vel_pad.max(),dvpix)
    iobs = (vel_obs_pad >= v_min) & (vel_obs_pad <= v_max)
    vel_lores = vel_obs_pad[iobs]
    nlores = vel_lores.size

    # original tau array was the stronger blue transition. Now shift velocity array to get tau array for red transition
    # tau_interp = interp1d(vel_pad, tau_igm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    tau_plot = tau_igm[:,iobs_hires]
    # tau_igm: total tau is the sum of the red and blue tau's

    tau_cgm = np.zeros_like(tau_igm)
    cgm_tuple = None

    # Compute the various fluxes with and without the CGM
    flux_tot_hires_pad = np.exp(-(tau_igm + tau_cgm))
    flux_igm_hires_pad = np.exp(-tau_igm)
    flux_cgm_hires_pad = np.exp(-tau_cgm)

    # Now smooth this and interpolate onto observational wavelength grid
    sigma_resolution = (fwhm/2.35483)/dvpix_hires  # fwhm = 2.3548 sigma
    flux_tot_sm = gaussian_filter1d(flux_tot_hires_pad, sigma_resolution, mode='mirror')
    flux_igm_sm = gaussian_filter1d(flux_igm_hires_pad, sigma_resolution, mode='mirror')
    flux_cgm_sm = gaussian_filter1d(flux_cgm_hires_pad, sigma_resolution, mode='mirror')

    flux_tot_interp = interp1d(vel_pad, flux_tot_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_igm_interp = interp1d(vel_pad, flux_igm_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_cgm_interp = interp1d(vel_pad, flux_cgm_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_tot_pad = flux_tot_interp(vel_obs_pad)
    flux_igm_pad = flux_igm_interp(vel_obs_pad)
    flux_cgm_pad = flux_cgm_interp(vel_obs_pad)

    # For the high-resolution spectrum take the valid region and recenter about zero
    flux_tot_hires = flux_tot_hires_pad[:,iobs_hires]
    flux_igm_hires = flux_igm_hires_pad[:,iobs_hires]
    flux_cgm_hires = flux_cgm_hires_pad[:,iobs_hires]

    oden = oden_pad[:,iobs_hires]
    T = T_pad[:,iobs_hires]
    v_los = v_los_pad[:,iobs_hires]

    # For the observed spectrum take the valid region and recenter about zero
    flux_tot_lores = flux_tot_pad[:,iobs]
    flux_igm_lores = flux_igm_pad[:,iobs]
    flux_cgm_lores = flux_cgm_pad[:,iobs]

    # Guarantee that nothing exceeds one due to roundoff error
    flux_tot_lores = np.clip(flux_tot_lores, None, 1.0)
    flux_igm_lores = np.clip(flux_igm_lores, None, 1.0)
    flux_cgm_lores = np.clip(flux_cgm_lores, None, 1.0)

    flux_tot_hires = np.clip(flux_tot_hires, None, 1.0)
    flux_igm_hires = np.clip(flux_igm_hires, None, 1.0)
    flux_cgm_hires = np.clip(flux_cgm_hires, None, 1.0)

    return vel_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), \
           vel_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), (oden, v_los, T), cgm_tuple, tau_plot


def create_metal_forest_tau(params, skewers, logZ, fwhm, metal_ion, z=None, sampling=3.0, cgm_dict=None, metal_dndz_func=None, seed=None, type = 'normal'):
    """
        Generate metal line forest at the specified metallicity, with the option to include CGM absorbers and output tau.

        Args:
            params (astropy table)
            skewers (astropy table)
            logZ (float)
            fwhm (float): in km/s
            metal_ion (str), e.g. 'C IV'. Requires whitespace between metal name and ionization stage
            z (float): redshift
            sampling (float): number of pixels per resolution element

            ::For incorporating CGM absorbers::
            cgm_dict (dictionary): containing parameters for the distribution of CGM absorbers.
            metal_dndz_func (function): used to compute dndz of CGM absorber and for drawing random W values. See create_metal_cgm().
            seed (int): random seed for drawing CGM absorbers.
            type: decide which forest is produced (red/blue/normal)
    """

    # ~0.00014 sec to generate one skewer
    if z is None:
        z = params['z'][0]

    dvpix = fwhm/sampling

    # Size of skewers and pixel scale at sim resolution
    vside = params['VSIDE'][0] # box size in km/s
    Ng = params['Ng'][0]       # number of grids on each side of the box
    dvpix_hires = vside/Ng     # pixel scale of the box in km/s
    nskew =len(skewers)

    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    tau0, f_ratio, v_metal, nh_bar = metal_tau0(metal_ion, z, logZ, cosmo=Planck15, X=0.76)
    # note that tau0 is obtained from the stronger blue line

    # Pad the skewer for the convolution
    npad = int(np.ceil((7.0*fwhm + v_metal.value)/dvpix_hires))
    v_pad = npad*dvpix_hires
    pad_tuple = ((0,0), (npad, npad))
    tau_blue = np.pad(tau0*skewers['TAU'].data, pad_tuple, 'wrap')

    xmetal_colname = 'X_' + metal_ion.replace(' ', '')
    xmetal_pad = np.pad(skewers[xmetal_colname].data, pad_tuple, 'wrap')

    oden_pad = np.pad(skewers['ODEN'].data, pad_tuple, 'wrap')
    T_pad = np.pad(skewers['T'].data, pad_tuple, 'wrap')
    v_los_pad = np.pad(skewers['VEL_Z'].data, pad_tuple, 'wrap')

    # Determine the velocity coverage including padding, etc.
    v_min = 0.0
    v_max = vside
    vel_pad = (v_min - v_pad) + np.arange(Ng + 2*npad)*dvpix_hires
    # For the high-resolution spectrum take the valid region and recenter about zero
    iobs_hires = (vel_pad >= v_min) & (vel_pad <= v_max)
    vel_hires = vel_pad[iobs_hires]
    nhires = vel_hires.size
    # For the observed spectrum take the valid region and recenter about zero
    vel_obs_pad = np.arange(vel_pad.min(),vel_pad.max(),dvpix)
    iobs = (vel_obs_pad >= v_min) & (vel_obs_pad <= v_max)
    vel_lores = vel_obs_pad[iobs]
    nlores = vel_lores.size

    # original tau array was the stronger blue transition. Now shift velocity array to get tau array for red transition
    tau_interp = interp1d(vel_pad + v_metal.value, tau_blue*f_ratio, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    tau_red = np.fmax(tau_interp(vel_pad), 0.0)
    if type == 'normal':
        tau_igm = tau_blue + tau_red # total tau is the sum of the red and blue tau's
    elif type == 'red':
        tau_igm = tau_red
    elif type == 'blue':
        tau_igm == tau_blue

    tau_interp = interp1d(vel_pad, tau_igm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    tau_plot = tau_igm[:,iobs_hires]
    # Now generate the CGM
    if cgm_dict != None:
        # tau_cgm, logN_draws, b_draws, v_draws, W_2796_draws, iskew_abs, tau_draws = create_mgii_cgm(vel_pad, nskew, z, cgm_dict, rand=rand)
        tau_cgm, logN_draws, b_draws, v_draws, W_blue_draws, iskew_abs, tau_draws = create_metal_cgm(vel_pad, nskew, z, cgm_dict, metal_dndz_func, metal_ion=metal_ion, seed=seed)

        # Only pass back the draws that reside in the final velocity coverage (as determined by vel_lores)
        ikeep = (v_draws > vel_lores.min()) & (v_draws < vel_lores.max())
        cgm_tuple = (logN_draws[ikeep], b_draws[ikeep], v_draws[ikeep], W_blue_draws[ikeep], iskew_abs[ikeep], tau_draws[ikeep, :])
    else:
        tau_cgm = np.zeros_like(tau_igm)
        cgm_tuple = None

    # Compute the various fluxes with and without the CGM
    flux_tot_hires_pad = np.exp(-(tau_igm + tau_cgm))
    flux_igm_hires_pad = np.exp(-tau_igm)
    flux_cgm_hires_pad = np.exp(-tau_cgm)

    # Now smooth this and interpolate onto observational wavelength grid
    sigma_resolution = (fwhm/2.35483)/dvpix_hires  # fwhm = 2.3548 sigma
    flux_tot_sm = gaussian_filter1d(flux_tot_hires_pad, sigma_resolution, mode='mirror')
    flux_igm_sm = gaussian_filter1d(flux_igm_hires_pad, sigma_resolution, mode='mirror')
    flux_cgm_sm = gaussian_filter1d(flux_cgm_hires_pad, sigma_resolution, mode='mirror')

    flux_tot_interp = interp1d(vel_pad, flux_tot_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_igm_interp = interp1d(vel_pad, flux_igm_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_cgm_interp = interp1d(vel_pad, flux_cgm_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    flux_tot_pad = flux_tot_interp(vel_obs_pad)
    flux_igm_pad = flux_igm_interp(vel_obs_pad)
    flux_cgm_pad = flux_cgm_interp(vel_obs_pad)

    # For the high-resolution spectrum take the valid region and recenter about zero
    flux_tot_hires = flux_tot_hires_pad[:,iobs_hires]
    flux_igm_hires = flux_igm_hires_pad[:,iobs_hires]
    flux_cgm_hires = flux_cgm_hires_pad[:,iobs_hires]

    x_metal = xmetal_pad[:,iobs_hires]
    oden = oden_pad[:,iobs_hires]
    T = T_pad[:,iobs_hires]
    v_los = v_los_pad[:,iobs_hires]

    # For the observed spectrum take the valid region and recenter about zero
    flux_tot_lores = flux_tot_pad[:,iobs]
    flux_igm_lores = flux_igm_pad[:,iobs]
    flux_cgm_lores = flux_cgm_pad[:,iobs]

    # Guarantee that nothing exceeds one due to roundoff error
    flux_tot_lores = np.clip(flux_tot_lores, None, 1.0)
    flux_igm_lores = np.clip(flux_igm_lores, None, 1.0)
    flux_cgm_lores = np.clip(flux_cgm_lores, None, 1.0)

    flux_tot_hires = np.clip(flux_tot_hires, None, 1.0)
    flux_igm_hires = np.clip(flux_igm_hires, None, 1.0)
    flux_cgm_hires = np.clip(flux_cgm_hires, None, 1.0)

    return vel_lores, (flux_tot_lores, flux_igm_lores, flux_cgm_lores), \
           vel_hires, (flux_tot_hires, flux_igm_hires, flux_cgm_hires), (oden, v_los, T, x_metal), cgm_tuple, tau_plot

def calc_igm_Zeff(fm, logZ_fid=-3.5):
    # calculates effective metallicity

    sol = labsol.SolarAbund()
    logZ_sol = sol.get_ratio('C/H') # same as sol['C'] - 12.0
    nC_nH_sol = 10**(logZ_sol)

    nH_bar = 3.1315263992114194e-05 # from skewerfile
    Z_fid = 10 ** (logZ_fid)
    nC_nH_fid = Z_fid * nC_nH_sol
    nC = nH_bar * nC_nH_fid * fm

    logZ_eff = np.log10(nC / nH_bar) - logZ_sol
    logZ_jfh = np.log10(10**(logZ_fid) * fm)

    return logZ_eff


def create_lya_forest_short(params, skewers, logZ, fwhm, metal_ion='C IV', z=None, sampling=3.0, cgm_dict=None, seed=None):

    # ~0.00014 sec to generate one skewer
    if z is None:
        z = params['z'][0]

    dvpix = fwhm/sampling

    # Size of skewers and pixel scale at sim resolution
    vside = params['VSIDE'][0] # box size in km/s
    Ng = params['Ng'][0]       # number of grids on each side of the box
    dvpix_hires = vside/Ng     # pixel scale of the box in km/s
    nskew =len(skewers)

    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    tau0, f_ratio, v_metal, nh_bar = metal_tau0(metal_ion, z, logZ, cosmo=Planck15, X=0.76)
    #tau0, nh_bar = lya_tau0(z, cosmo=Planck15, X=0.76)
    # note that tau0 is obtained from the stronger blue line

    # Pad the skewer for the convolution
    npad = int(np.ceil((7.0*fwhm + v_metal.value)/dvpix_hires)) ## v_metal = 0 and f_ratio = 1 is possible?
    v_pad = npad*dvpix_hires
    pad_tuple = ((0,0), (npad, npad))
    #tau_igm = np.pad(tau0*skewers['TAU'].data, pad_tuple, 'wrap')
    tau_igm = np.pad(skewers['TAU'].data, pad_tuple, 'wrap')

    # Determine the velocity coverage including padding, etc.
    v_min = 0.0
    v_max = vside
    vel_pad = (v_min - v_pad) + np.arange(Ng + 2*npad)*dvpix_hires
    # For the high-resolution spectrum take the valid region and recenter about zero
    iobs_hires = (vel_pad >= v_min) & (vel_pad <= v_max)
    vel_hires = vel_pad[iobs_hires]
    nhires = vel_hires.size
    # For the observed spectrum take the valid region and recenter about zero
    vel_obs_pad = np.arange(vel_pad.min(),vel_pad.max(),dvpix)
    iobs = (vel_obs_pad >= v_min) & (vel_obs_pad <= v_max)
    vel_lores = vel_obs_pad[iobs]
    nlores = vel_lores.size

    tau_cgm = np.zeros_like(tau_igm)
    cgm_tuple = None

    # Compute the various fluxes with and without the CGM
    flux_tot_hires_pad = np.exp(-(tau_igm + tau_cgm))

    # Now smooth this and interpolate onto observational wavelength grid
    sigma_resolution = (fwhm/2.35483)/dvpix_hires  # fwhm = 2.3548 sigma

    flux_tot_sm = gaussian_filter1d(flux_tot_hires_pad, sigma_resolution, mode='mirror')

    flux_tot_interp = interp1d(vel_pad, flux_tot_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)

    flux_tot_pad = flux_tot_interp(vel_obs_pad)

    # For the observed spectrum take the valid region and recenter about zero
    flux_tot_lores = flux_tot_pad[:,iobs]

    # Guarantee that nothing exceeds one due to roundoff error
    flux_tot_lores = np.clip(flux_tot_lores, None, 1.0)

    return vel_lores, flux_tot_lores

def create_metal_forest_short(params, skewers, logZ, fwhm, metal_ion, z=None, sampling=3.0, cgm_dict=None, metal_dndz_func=None, seed=None):


    # ~0.00014 sec to generate one skewer
    if z is None:
        z = params['z'][0]

    dvpix = fwhm/sampling

    # Size of skewers and pixel scale at sim resolution
    vside = params['VSIDE'][0] # box size in km/s
    Ng = params['Ng'][0]       # number of grids on each side of the box
    dvpix_hires = vside/Ng     # pixel scale of the box in km/s
    nskew =len(skewers)

    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    tau0, f_ratio, v_metal, nh_bar = metal_tau0(metal_ion, z, logZ, cosmo=Planck15, X=0.76)
    # note that tau0 is obtained from the stronger blue line

    # Pad the skewer for the convolution
    npad = int(np.ceil((7.0*fwhm + v_metal.value)/dvpix_hires))
    v_pad = npad*dvpix_hires
    pad_tuple = ((0,0), (npad, npad))
    tau_blue = np.pad(tau0*skewers['TAU'].data, pad_tuple, 'wrap')

    xmetal_colname = 'X_' + metal_ion.replace(' ', '')
    xmetal_pad = np.pad(skewers[xmetal_colname].data, pad_tuple, 'wrap')

    # Determine the velocity coverage including padding, etc.
    v_min = 0.0
    v_max = vside
    vel_pad = (v_min - v_pad) + np.arange(Ng + 2*npad)*dvpix_hires
    # For the high-resolution spectrum take the valid region and recenter about zero
    iobs_hires = (vel_pad >= v_min) & (vel_pad <= v_max)
    vel_hires = vel_pad[iobs_hires]
    nhires = vel_hires.size
    # For the observed spectrum take the valid region and recenter about zero
    vel_obs_pad = np.arange(vel_pad.min(),vel_pad.max(),dvpix)
    iobs = (vel_obs_pad >= v_min) & (vel_obs_pad <= v_max)
    vel_lores = vel_obs_pad[iobs]
    nlores = vel_lores.size

    # original tau array was the stronger blue transition. Now shift velocity array to get tau array for red transition
    tau_interp = interp1d(vel_pad + v_metal.value, tau_blue*f_ratio, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    tau_red = np.fmax(tau_interp(vel_pad), 0.0)
    tau_igm = tau_blue + tau_red # total tau is the sum of the red and blue tau's

    tau_interp = interp1d(vel_pad, tau_igm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    tau_plot = tau_igm[:,iobs_hires]
    # Now generate the CGM
    if cgm_dict != None:
        # tau_cgm, logN_draws, b_draws, v_draws, W_2796_draws, iskew_abs, tau_draws = create_mgii_cgm(vel_pad, nskew, z, cgm_dict, rand=rand)
        tau_cgm, logN_draws, b_draws, v_draws, W_blue_draws, iskew_abs, tau_draws = create_metal_cgm(vel_pad, nskew, z, cgm_dict, metal_dndz_func, metal_ion=metal_ion, seed=seed)

        # Only pass back the draws that reside in the final velocity coverage (as determined by vel_lores)
        ikeep = (v_draws > vel_lores.min()) & (v_draws < vel_lores.max())
        cgm_tuple = (logN_draws[ikeep], b_draws[ikeep], v_draws[ikeep], W_blue_draws[ikeep], iskew_abs[ikeep], tau_draws[ikeep, :])
    else:
        tau_cgm = np.zeros_like(tau_igm)
        cgm_tuple = None

    # Compute the various fluxes with and without the CGM
    flux_tot_hires_pad = np.exp(-(tau_igm + tau_cgm))

    # Now smooth this and interpolate onto observational wavelength grid
    sigma_resolution = (fwhm/2.35483)/dvpix_hires  # fwhm = 2.3548 sigma
    flux_tot_sm = gaussian_filter1d(flux_tot_hires_pad, sigma_resolution, mode='mirror')

    flux_tot_interp = interp1d(vel_pad, flux_tot_sm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)

    flux_tot_pad = flux_tot_interp(vel_obs_pad)

    # For the observed spectrum take the valid region and recenter about zero
    flux_tot_lores = flux_tot_pad[:,iobs]

    # Guarantee that nothing exceeds one due to roundoff error
    flux_tot_lores = np.clip(flux_tot_lores, None, 1.0)

    return vel_lores, flux_tot_lores


def imap_unordered_bar(func, args, nproc):
    """
    Display progress bar.
    """
    p = Pool(processes=nproc)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

def interp_likelihood_covar_nproc(init_out, nlogM_fine, nR_fine, nlogZ_fine, nproc=5, interpolate_covar=True):

    # unpack input
    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, logM_guess, R_guess, logZ_guess = init_out

    # Interpolate the likelihood onto a fine grid to speed up the MCMC

    nlogM = logM_coarse.size
    logM_fine_min = logM_coarse.min()
    logM_fine_max = logM_coarse.max()
    dlogM_fine = (logM_fine_max - logM_fine_min) / (nlogM_fine - 1)
    logM_fine = logM_fine_min + np.arange(nlogM_fine) * dlogM_fine
    logM_fine[-1] = logM_coarse[-1]
    logM_fine[0] = logM_coarse[0]

    nR = R_coarse.size
    R_fine_min = R_coarse.min()
    R_fine_max = R_coarse.max()
    dR_fine = (R_fine_max - R_fine_min) / (nR_fine - 1)
    R_fine = R_fine_min + np.arange(nR_fine) * dR_fine
    R_fine[-1] = R_coarse[-1]
    R_fine[0] = R_coarse[0]

    nlogZ = logZ_coarse.size
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine
    logZ_fine[-1] = logZ_coarse[-1]
    logZ_fine[0] = logZ_coarse[0]

    logM_fine_unit = []
    R_fine_unit = []
    logZ_fine_unit = []

    logM_fine_unit_loc = []
    R_fine_unit_loc = []
    logZ_fine_unit_loc = []

    for i in range(nlogM-1):
        if i == nlogM-2:
            logM_fine_unit.append(logM_fine[np.where((logM_fine >= logM_coarse[i]) & (logM_fine <= logM_coarse[i+1]))[0]])
            logM_fine_unit_loc.append(np.where((logM_fine >= logM_coarse[i]) & (logM_fine <= logM_coarse[i+1]))[0])
        else:
            logM_fine_unit.append(logM_fine[np.where((logM_fine >= logM_coarse[i]) & (logM_fine < logM_coarse[i+1]))[0]])
            logM_fine_unit_loc.append(np.where((logM_fine >= logM_coarse[i]) & (logM_fine < logM_coarse[i+1]))[0])

    for i in range(nR-1):
        if i == nR-2:
            R_fine_unit.append(R_fine[np.where((R_fine >= R_coarse[i]) & (R_fine <= R_coarse[i+1]))[0]])
            R_fine_unit_loc.append(np.where((R_fine >= R_coarse[i]) & (R_fine <= R_coarse[i+1]))[0])
        else:
            R_fine_unit.append(R_fine[np.where((R_fine >= R_coarse[i]) & (R_fine < R_coarse[i+1]))[0]])
            R_fine_unit_loc.append(np.where((R_fine >= R_coarse[i]) & (R_fine < R_coarse[i+1]))[0])


    for i in range(nlogZ-1):
        if i == nlogZ-2:
            logZ_fine_unit.append(logZ_fine[np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine <= logZ_coarse[i+1]))[0]])
            logZ_fine_unit_loc.append(np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine <= logZ_coarse[i+1]))[0])
        else:
            logZ_fine_unit.append(logZ_fine[np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine < logZ_coarse[i+1]))[0]])
            logZ_fine_unit_loc.append(np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine < logZ_coarse[i+1]))[0])

    logM_fine_unit = np.array(logM_fine_unit, dtype='object')
    R_fine_unit = np.array(R_fine_unit, dtype='object')
    logZ_fine_unit = np.array(logZ_fine_unit, dtype='object')

    logM_fine_unit_loc = np.array(logM_fine_unit_loc, dtype='object')
    R_fine_unit_loc = np.array(R_fine_unit_loc, dtype='object')
    logZ_fine_unit_loc = np.array(logZ_fine_unit_loc, dtype='object')

    print('dlogM_fine', dlogM_fine)
    print('dR_fine' %dR_fine)
    print('dlogZ_fine', dlogZ_fine)

    lnlike_fine = np.zeros((nlogM_fine, nR_fine, nlogZ_fine))
    xi_model_fine = np.zeros((nlogM_fine, nR_fine, nlogZ_fine, xi_model_array.shape[-1]))
    all_args = []

    for i in range(nlogM-1):
        for j in range(nR-1):
            for k in range(nlogZ-1):
                itup = (i, j, k, logM_fine_unit[i], R_fine_unit[j], logZ_fine_unit[k], \
                logM_fine_unit_loc[i], R_fine_unit_loc[j], logZ_fine_unit_loc[k],\
                logM_coarse[i:i+2], R_coarse[j:j+2], logZ_coarse[k:k+2],\
                xi_model_array[i:i+2,j:j+2,k:k+2,:], lndet_array[i:i+2,j:j+2,k:k+2], covar_array[i:i+2,j:j+2,k:k+2,:,:],\
                xi_data, xi_mask)
                all_args.append(itup)

    output = imap_unordered_bar(likelihood_covar_calc, all_args, nproc)
    for out in output:
        nlogM, nR, nlogZ, logM_fine_unit_loc, R_fine_unit_loc, logZ_fine_unit_loc, lnlike_unit, xi_unit = out
        for i in logM_fine_unit_loc:
            for j in R_fine_unit_loc:
                for k in logZ_fine_unit_loc:
                    lnlike_fine[i, j, k] = lnlike_unit[np.where(logM_fine_unit_loc==i)[0][0], np.where(R_fine_unit_loc==j)[0][0], np.where(logZ_fine_unit_loc==k)[0][0]]
                    xi_model_fine[i, j, k, :] = xi_unit[np.where(logM_fine_unit_loc==i)[0][0], np.where(R_fine_unit_loc==j)[0][0], np.where(logZ_fine_unit_loc==k)[0][0], :]
    logM_max, R_max, logZ_max = np.where(lnlike_fine==lnlike_fine.max())

    print('The most possible grid in fine_cov is logM = %.2f, R = %.2f and logZ = %.2f' % (logM_fine[logM_max], R_fine[R_max], logZ_fine[logZ_max]))

    return lnlike_fine, xi_model_fine, logM_fine, R_fine, logZ_fine

def likelihood_covar_calc(args):

    nlogM, nR, nlogZ, logM_fine, R_fine, logZ_fine, \
    logM_fine_unit_loc, R_fine_unit_loc, logZ_fine_unit_loc,\
    logM_coarse, R_coarse, logZ_coarse, xi_model_array, lndet_array, covar_array, xi_data, xi_mask = args

    xi_model_fine, lndet_fine, covar_fine = inference.interp_model_all(logM_fine, R_fine, \
    logZ_fine, logM_coarse, R_coarse, logZ_coarse, xi_model_array, lndet_array, covar_array)

    lnlike_fine = np.zeros((len(logM_fine), len(R_fine), len(logZ_fine)))
    for ilogM, logM_val in enumerate(logM_fine):
        for iR, R_val in enumerate(R_fine):
            for ilogZ, logZ_val in enumerate(logZ_fine):
                lnlike_fine[ilogM, iR, ilogZ] = inference.lnlike_calc(xi_data, xi_mask,
                                                                        xi_model_fine[ilogM, iR, ilogZ, :],
                                                                        lndet_fine[ilogM, iR, ilogZ],
                                                                        covar_fine[ilogM, iR, ilogZ, :, :])

    return nlogM, nR, nlogZ, logM_fine_unit_loc, R_fine_unit_loc, logZ_fine_unit_loc, lnlike_fine, xi_model_fine


def covar_fine_generator(init_out, nlogM_fine, nR_fine, nlogZ_fine, covar_path, nproc=5, interpolate_covar=True):

    # unpack input
    logM_coarse, R_coarse, logZ_coarse, xi_model_array, covar_array, lndet_array = init_out

    # Interpolate the likelihood onto a fine grid to speed up the MCMC

    nlogM = logM_coarse.size
    logM_fine_min = logM_coarse.min()
    logM_fine_max = logM_coarse.max()
    dlogM_fine = (logM_fine_max - logM_fine_min) / (nlogM_fine - 1)
    logM_fine = logM_fine_min + np.arange(nlogM_fine) * dlogM_fine
    logM_fine[-1] = logM_coarse[-1]
    logM_fine[0] = logM_coarse[0]

    nR = R_coarse.size
    R_fine_min = R_coarse.min()
    R_fine_max = R_coarse.max()
    dR_fine = (R_fine_max - R_fine_min) / (nR_fine - 1)
    R_fine = R_fine_min + np.arange(nR_fine) * dR_fine
    R_fine[-1] = R_coarse[-1]
    R_fine[0] = R_coarse[0]

    nlogZ = logZ_coarse.size
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine
    logZ_fine[-1] = logZ_coarse[-1]
    logZ_fine[0] = logZ_coarse[0]

    logM_fine_unit = []
    R_fine_unit = []
    logZ_fine_unit = []

    logM_fine_unit_loc = []
    R_fine_unit_loc = []
    logZ_fine_unit_loc = []

    for i in range(nlogM-1):
        if i == nlogM-2:
            logM_fine_unit.append(logM_fine[np.where((logM_fine >= logM_coarse[i]) & (logM_fine <= logM_coarse[i+1]))[0]])
            logM_fine_unit_loc.append(np.where((logM_fine >= logM_coarse[i]) & (logM_fine <= logM_coarse[i+1]))[0])
        else:
            logM_fine_unit.append(logM_fine[np.where((logM_fine >= logM_coarse[i]) & (logM_fine < logM_coarse[i+1]))[0]])
            logM_fine_unit_loc.append(np.where((logM_fine >= logM_coarse[i]) & (logM_fine < logM_coarse[i+1]))[0])

    for i in range(nR-1):
        if i == nR-2:
            R_fine_unit.append(R_fine[np.where((R_fine >= R_coarse[i]) & (R_fine <= R_coarse[i+1]))[0]])
            R_fine_unit_loc.append(np.where((R_fine >= R_coarse[i]) & (R_fine <= R_coarse[i+1]))[0])
        else:
            R_fine_unit.append(R_fine[np.where((R_fine >= R_coarse[i]) & (R_fine < R_coarse[i+1]))[0]])
            R_fine_unit_loc.append(np.where((R_fine >= R_coarse[i]) & (R_fine < R_coarse[i+1]))[0])

    for i in range(nlogZ-1):
        if i == nlogZ-2:
            logZ_fine_unit.append(logZ_fine[np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine <= logZ_coarse[i+1]))[0]])
            logZ_fine_unit_loc.append(np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine <= logZ_coarse[i+1]))[0])
        else:
            logZ_fine_unit.append(logZ_fine[np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine < logZ_coarse[i+1]))[0]])
            logZ_fine_unit_loc.append(np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine < logZ_coarse[i+1]))[0])

    logM_fine_unit = np.array(logM_fine_unit, dtype='object')
    R_fine_unit = np.array(R_fine_unit, dtype='object')
    logZ_fine_unit = np.array(logZ_fine_unit, dtype='object')

    logM_fine_unit_loc = np.array(logM_fine_unit_loc, dtype='object')
    R_fine_unit_loc = np.array(R_fine_unit_loc, dtype='object')
    logZ_fine_unit_loc = np.array(logZ_fine_unit_loc, dtype='object')

    print('dlogM_fine', dlogM_fine)
    print('dR_fine', dR_fine)
    print('dlogZ_fine', dlogZ_fine)

    all_args = []

    for i in range(nlogM-1):
        for j in range(nR-1):
            for k in range(nlogZ-1):
                itup = (i, j, k, logM_fine_unit[i], R_fine_unit[j], logZ_fine_unit[k], \
                logM_fine_unit_loc[i], R_fine_unit_loc[j], logZ_fine_unit_loc[k],\
                logM_coarse[i:i+2], R_coarse[j:j+2], logZ_coarse[k:k+2],\
                xi_model_array[i:i+2,j:j+2,k:k+2,:], lndet_array[i:i+2,j:j+2,k:k+2], covar_array[i:i+2,j:j+2,k:k+2,:,:], covar_path)
                all_args.append(itup)

    output = imap_unordered_bar(covar_fine_unit, all_args, nproc)

    return 0

def covar_fine_unit(args):

    nlogM, nR, nlogZ, logM_fine, R_fine, logZ_fine, \
    logM_fine_unit_loc, R_fine_unit_loc, logZ_fine_unit_loc,\
    logM_coarse, R_coarse, logZ_coarse, xi_model_array, lndet_array, covar_array, covar_path = args

    xi_model_fine, lndet_fine, covar_fine = inference.interp_model_all(logM_fine, R_fine, \
    logZ_fine, logM_coarse, R_coarse, logZ_coarse, xi_model_array, lndet_array, covar_array)

    with open(covar_path + 'covariance_nlogM_%.2f_nR_%.2f_nlogZ_%.2f.npy' % (nlogM, nR, nlogZ), 'wb') as f:
        np.save(f, xi_model_fine)
        np.save(f, lndet_fine)
        np.save(f, covar_fine)

    return 0


def inference_test_CIV_lya(nsteps, burnin, nwalkers, logM_fine, R_fine, logZ_fine, lnlike_fine, linear_prior, ball_size=0.01, \
                   seed=None, nproc = 50, test_num = 1000):

    all_args = []
    logM = []
    R = []
    logZ = []
    for i in range(test_num):
        itup = (nsteps, burnin, nwalkers, logM_fine, R_fine, logZ_fine, lnlike_fine[i], linear_prior, ball_size, seed)
        all_args.append(itup)
    output = imap_unordered_bar(inference_unit, all_args, nproc)
    for out in output:
        logM_infer, R_infer, logZ_infer = out
        logM.append(logM_infer)
        R.append(R_infer)
        logZ.append(logZ_infer)

    return logM, R, logZ


def inference_unit(args):

    nsteps, burnin, nwalkers, logM_fine, R_fine, logZ_fine, lnlike_fine, linear_prior, ball_size, seed = args

    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    if seed == None:
        seed = np.random.randint(0, 10000000)
        print("Using random seed", seed)
    else:
        print("Using random seed", seed)

    print("Using ball size", ball_size)

    rand = np.random.RandomState(seed)

    logM_fine_min, logM_fine_max = logM_fine.min(), logM_fine.max()
    R_fine_min, R_fine_max = R_fine.min(), R_fine.max()
    logZ_fine_min, logZ_fine_max = logZ_fine.min(), logZ_fine.max()

    bounds = [(logM_fine_min, logM_fine_max), (R_fine_min, R_fine_max), (logZ_fine_min, logZ_fine_max)]

    chi2_func = lambda *args: -2 * inference.lnprob_3d(*args)
    args = (lnlike_fine, logM_fine, R_fine, logZ_fine, linear_prior)

    result_opt = optimize.differential_evolution(chi2_func, bounds=bounds, popsize=25, recombination=0.7, disp=True, polish=True, args=args, seed=rand)
    ndim = 3
    # initialize walkers
    # for my own understanding #
    pos = []
    for i in range(nwalkers):
        tmp = []
        for j in range(ndim):
            perturb_pos = result_opt.x[j] + (ball_size * (bounds[j][1] - bounds[j][0]) * rand.randn(1)[0])
            tmp.append(np.clip(perturb_pos, bounds[j][0], bounds[j][1]))
        pos.append(tmp)

    pos = [[np.clip(result_opt.x[i] + 1e-2 * (bounds[i][1] - bounds[i][0]) * rand.randn(1)[0], bounds[i][0], bounds[i][1])
        for i in range(ndim)] for i in range(nwalkers)]

    # np.random.seed(rand.randint(0, seed, size=1)[0])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, inference.lnprob_3d, args = args)
    sampler.run_mcmc(pos, nsteps, progress=True)

    tau = sampler.get_autocorr_time()
    print('Autocorrelation time')
    print('tau_logM = {:7.2f}, tau_R = {:7.2f}, tau_logZ = {:7.2f}'.format(tau[0], tau[1], tau[2]))

    param_samples = sampler.get_chain(discard=burnin, thin=250, flat=True) # numpy array

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    param = np.median(param_samples, axis=0)

    logM_infer = param[0]
    R_infer = param[1]
    logZ_infer = param[2]

    return logM_infer, R_infer, logZ_infer


def interp_likelihood_inference_test(init_out, nlogM_fine, nR_fine, nlogZ_fine, covar_path, nproc=5):

    # unpack input
    logM_data, R_data, logZ_data, xi_data, xi_mask = init_out

    logZ_coarse = np.linspace(-4.5, -2.0, 26)
    logM_coarse = np.arange(8.5, 11.0+0.1, 0.1)
    R_coarse = np.arange(0.1, 3.0+0.1, 0.1)

    # Interpolate the likelihood onto a fine grid to speed up the MCMC

    nlogM = logM_coarse.size
    logM_fine_min = logM_coarse.min()
    logM_fine_max = logM_coarse.max()
    dlogM_fine = (logM_fine_max - logM_fine_min) / (nlogM_fine - 1)
    logM_fine = logM_fine_min + np.arange(nlogM_fine) * dlogM_fine
    logM_fine[-1] = logM_coarse[-1]
    logM_fine[0] = logM_coarse[0]

    nR = R_coarse.size
    R_fine_min = R_coarse.min()
    R_fine_max = R_coarse.max()
    dR_fine = (R_fine_max - R_fine_min) / (nR_fine - 1)
    R_fine = R_fine_min + np.arange(nR_fine) * dR_fine
    R_fine[-1] = R_coarse[-1]
    R_fine[0] = R_coarse[0]

    nlogZ = logZ_coarse.size
    logZ_fine_min = logZ_coarse.min()
    logZ_fine_max = logZ_coarse.max()
    dlogZ_fine = (logZ_fine_max - logZ_fine_min) / (nlogZ_fine - 1)
    logZ_fine = logZ_fine_min + np.arange(nlogZ_fine) * dlogZ_fine
    logZ_fine[-1] = logZ_coarse[-1]
    logZ_fine[0] = logZ_coarse[0]

    logM_fine_unit = []
    R_fine_unit = []
    logZ_fine_unit = []

    logM_fine_unit_loc = []
    R_fine_unit_loc = []
    logZ_fine_unit_loc = []

    for i in range(nlogM-1):
        if i == nlogM-2:
            logM_fine_unit.append(logM_fine[np.where((logM_fine >= logM_coarse[i]) & (logM_fine <= logM_coarse[i+1]))[0]])
            logM_fine_unit_loc.append(np.where((logM_fine >= logM_coarse[i]) & (logM_fine <= logM_coarse[i+1]))[0])
        else:
            logM_fine_unit.append(logM_fine[np.where((logM_fine >= logM_coarse[i]) & (logM_fine < logM_coarse[i+1]))[0]])
            logM_fine_unit_loc.append(np.where((logM_fine >= logM_coarse[i]) & (logM_fine < logM_coarse[i+1]))[0])

    for i in range(nR-1):
        if i == nR-2:
            R_fine_unit.append(R_fine[np.where((R_fine >= R_coarse[i]) & (R_fine <= R_coarse[i+1]))[0]])
            R_fine_unit_loc.append(np.where((R_fine >= R_coarse[i]) & (R_fine <= R_coarse[i+1]))[0])
        else:
            R_fine_unit.append(R_fine[np.where((R_fine >= R_coarse[i]) & (R_fine < R_coarse[i+1]))[0]])
            R_fine_unit_loc.append(np.where((R_fine >= R_coarse[i]) & (R_fine < R_coarse[i+1]))[0])


    for i in range(nlogZ-1):
        if i == nlogZ-2:
            logZ_fine_unit.append(logZ_fine[np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine <= logZ_coarse[i+1]))[0]])
            logZ_fine_unit_loc.append(np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine <= logZ_coarse[i+1]))[0])
        else:
            logZ_fine_unit.append(logZ_fine[np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine < logZ_coarse[i+1]))[0]])
            logZ_fine_unit_loc.append(np.where((logZ_fine >= logZ_coarse[i]) & (logZ_fine < logZ_coarse[i+1]))[0])

    logM_fine_unit = np.array(logM_fine_unit, dtype='object')
    R_fine_unit = np.array(R_fine_unit, dtype='object')
    logZ_fine_unit = np.array(logZ_fine_unit, dtype='object')

    logM_fine_unit_loc = np.array(logM_fine_unit_loc, dtype='object')
    R_fine_unit_loc = np.array(R_fine_unit_loc, dtype='object')
    logZ_fine_unit_loc = np.array(logZ_fine_unit_loc, dtype='object')

    print('dlogM_fine', dlogM_fine)
    print('dR_fine' %dR_fine)
    print('dlogZ_fine', dlogZ_fine)

    lnlike_fine = np.zeros((nlogM_fine, nR_fine, nlogZ_fine))
    all_args = []

    for i in range(nlogM-1):
        for j in range(nR-1):
            for k in range(nlogZ-1):
                itup = (i, j, k, logM_fine_unit[i], R_fine_unit[j], logZ_fine_unit[k], \
                logM_fine_unit_loc[i], R_fine_unit_loc[j], logZ_fine_unit_loc[k],\
                logM_coarse[i:i+2], R_coarse[j:j+2], logZ_coarse[k:k+2],\
                xi_data, xi_mask, covar_path)
                all_args.append(itup)

    output = imap_unordered_bar(likelihood_calc, all_args, nproc)
    for out in output:
        nlogM, nR, nlogZ, logM_fine_unit_loc, R_fine_unit_loc, logZ_fine_unit_loc, lnlike_unit = out
        for i in logM_fine_unit_loc:
            for j in R_fine_unit_loc:
                for k in logZ_fine_unit_loc:
                    lnlike_fine[i, j, k] = lnlike_unit[np.where(logM_fine_unit_loc==i)[0][0], np.where(R_fine_unit_loc==j)[0][0], np.where(logZ_fine_unit_loc==k)[0][0]]
#                    xi_model_fine[i, j, k, :] = xi_unit[np.where(logM_fine_unit_loc==i)[0][0], np.where(R_fine_unit_loc==j)[0][0], np.where(logZ_fine_unit_loc==k)[0][0], :]
#    logM_max, R_max, logZ_max = np.where(lnlike_fine==lnlike_fine.max())

#    print('The most possible grid in fine_cov is logM = %.2f, R = %.2f and logZ = %.2f' % (logM_fine[logM_max], R_fine[R_max], logZ_fine[logZ_max]))

    return lnlike_fine, logM_fine, R_fine, logZ_fine

def likelihood_calc(args):

    nlogM, nR, nlogZ, logM_fine, R_fine, logZ_fine, \
    logM_fine_unit_loc, R_fine_unit_loc, logZ_fine_unit_loc,\
    logM_coarse, R_coarse, logZ_coarse, xi_data, xi_mask, covar_path = args

    with open(covar_path + 'covariance_nlogM_%.2f_nR_%.2f_nlogZ_%.2f.npy' % (nlogM, nR, nlogZ), 'rb') as f:
        xi_model_fine = np.load(f)
        lndet_fine = np.load(f)
        covar_fine = np.load(f)

    lnlike_fine = np.zeros((len(logM_fine), len(R_fine), len(logZ_fine)))
    for ilogM, logM_val in enumerate(logM_fine):
        for iR, R_val in enumerate(R_fine):
            for ilogZ, logZ_val in enumerate(logZ_fine):
                lnlike_fine[ilogM, iR, ilogZ] = inference.lnlike_calc(xi_data, xi_mask,
                                                                        xi_model_fine[ilogM, iR, ilogZ, :],
                                                                        lndet_fine[ilogM, iR, ilogZ],
                                                                        covar_fine[ilogM, iR, ilogZ, :, :])

    return nlogM, nR, nlogZ, logM_fine_unit_loc, R_fine_unit_loc, logZ_fine_unit_loc, lnlike_fine



def fv_logZ_eff_grid(param_samples, fvfm_file):

    logM_grid_coarse = np.linspace(8.5, 11.0, 26)
    R_grid_coarse = np.linspace(0.1, 3.0, 30)

    fv_coarse = np.zeros((len(logM_grid_coarse),len(R_grid_coarse)))
    fm_coarse = np.zeros((len(logM_grid_coarse),len(R_grid_coarse)))

    for i,logM in enumerate(logM_grid_coarse):
        for j, R in enumerate(R_grid_coarse):
            fv_coarse[i,j], fm_coarse[i,j] = halos_skewers.get_fvfm(np.round(logM, 2), np.round(R, 2), fvfm_file=fvfm_file)

    logM_array = param_samples[:,0]
    R_array = param_samples[:,1]
    logZ_array = param_samples[:,2]

    fv_array = []
    logZ_eff_array = []

    fv_func = RegularGridInterpolator((logM_grid_coarse, R_grid_coarse), fv_coarse)
    fm_func = RegularGridInterpolator((logM_grid_coarse, R_grid_coarse), fm_coarse)

    fv_out = fv_func(param_samples[:,0:2])
    fm_out = fm_func(param_samples[:,0:2])

    for i in range(len(fv_out)):
        fv = fv_out[i]
        fm = fm_out[i]
        logZ = logZ_array[i]
        logZ_eff = calc_igm_Zeff(fm, logZ)

        fv_array.append(fv)
        logZ_eff_array.append(logZ_eff)

    param_samples_new = np.column_stack((np.array(fv_array), np.array(logZ_eff_array)))

    return param_samples_new

def plot_mcmc_fv_logZ_eff(sampler, param_samples, param_samples_new, init_out, params, logM_fine, R_fine, logZ_fine, xi_model_fine, linear_prior, outpath_local, seed=None, overplot=False, overplot_param=None, fvfm_file=None):
    # seed here used to choose random nrand(=50) mcmc realizations to plot on the 2PCF measurement

    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, logM_guess, R_guess, logZ_guess = init_out

    ##### (1) Make the walker plot, use the true values in the chain
    var_label = ['fv', 'logZ_eff']

    fv_data, fm_data = halos_skewers.get_fvfm(np.round(logM_data, 2), np.round(R_data, 2),fvfm_file=fvfm_file)
    logZ_eff_data = calc_igm_Zeff(fm_data, logZ_data)

    #truths = [10**(logM_data), R_data, 10**(logZ_data)] if linear_prior else [logM_data, R_data, logZ_data]
    truths = [fv_data, logZ_eff_data] # (8/16/21) linear_prior only on logZ
    print("truths", truths)

    print(param_samples.shape)
    ##### (2) Make the corner plot, again use the true values in the chain
    fig = corner.corner(param_samples_new, labels=var_label, range=[(0,1),(-4.5,-2.0)], truths=truths, levels=(0.68, 0.95, 0.997), color='k', \
                        truth_color='darkgreen', \
                        show_titles=True, title_kwargs={"fontsize": 15}, label_kwargs={'fontsize': 20}, \
                        data_kwargs={'ms': 1.0, 'alpha': 0.1})
    if overplot == True:
        corner.corner(overplot_param, fig=fig, color='r')
    for ax in fig.get_axes():
        # ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.tick_params(labelsize=12)

    plt.savefig(outpath_local + 'corner_fv_logZ.pdf')
    plt.close()

    ##### (3) Make the corrfunc plot with mcmc realizations
    if fvfm_file != None:
        fv, fm = halos_skewers.get_fvfm(np.round(logM_data,2), np.round(R_data,2), fvfm_file=fvfm_file)
    else:
        fv, fm = halos_skewers.get_fvfm(np.round(logM_data,2), np.round(R_data,2))
    logZ_eff = halos_skewers.calc_igm_Zeff(fm, logZ_fid=logZ_data)
    print("logZ_eff", logZ_eff)
    corrfunc_plot_3d_fv_logZ_eff(xi_data, param_samples, param_samples_new, params, logM_fine, R_fine, logZ_fine, xi_model_fine, logM_coarse, R_coarse,
                     logZ_coarse, covar_array, fv_data, logZ_eff_data, outpath_local, nrand=50, seed=seed, fvfm_file=fvfm_file)


def corrfunc_plot_3d_fv_logZ_eff(xi_data, samples, samples_new, params, logM_fine, R_fine, logZ_fine, xi_model_fine, logM_coarse, R_coarse, logZ_coarse, \
                     covar_array, fv_data, logZ_eff_data, outpath_local, nrand=50, seed=None, fvfm_file=None):

    if seed == None:
        seed = np.random.randint(0, 10000000)
        print("Using random seed", seed)
    else:
        print("Using random seed", seed)

    rand = np.random.RandomState(seed)
    factor = 1000

    vel_corr = params['vel_mid'].flatten()
    vel_min = params['vmin_corr']
    vel_max = params['vmax_corr']

    # Compute the mean model from the samples
    xi_model_samp = inference.xi_model_3d(samples, logM_fine, R_fine, logZ_fine, xi_model_fine)
    xi_model_samp_mean = np.mean(xi_model_samp, axis=0)

    # Compute the covariance at the mean model
    theta_mean = np.mean(samples, axis=0)
    covar_mean = inference.covar_model_3d(theta_mean, logM_coarse, R_coarse, logZ_coarse, covar_array)
    xi_err = np.sqrt(np.diag(covar_mean))

    # Grab some realizations
    imock = rand.choice(np.arange(samples.shape[0]), size=nrand)
    xi_model_rand = xi_model_samp[imock, :]
    ymin = factor * np.min(xi_data - 1.3 * xi_err)
    ymax = factor * np.max(xi_data + 1.6 * xi_err)
    #ymax = 2*factor * np.max(xi_data + 1.6 * xi_err)

    # Plotting
    fx = plt.figure(1, figsize=(12, 7))
    # left, bottom, width, height
    rect = [0.12, 0.12, 0.84, 0.75]
    axis = fx.add_axes(rect)

    axis.errorbar(vel_corr, factor*xi_data, yerr=factor*xi_err, marker='o', ms=6, color='black', ecolor='black', capthick=2,
                  capsize=4, alpha=0.8, mec='none', ls='none', label='mock data', zorder=20)
    axis.plot(vel_corr, factor*xi_model_samp_mean, linewidth=2.0, color='red', zorder=10, label='inferred model')

    axis.set_xlabel(r'$\Delta v$ [km/s]', fontsize=26)
    #axis.set_ylabel(r'$\xi(\Delta v)$', fontsize=26, labelpad=-4)
    axis.set_ylabel(r'$\xi(\Delta v)\times %d$' % factor, fontsize=26, labelpad=-4)

    axis.tick_params(axis="x", labelsize=16)
    axis.tick_params(axis="y", labelsize=16)

    #xoffset = -0.1
    #offset = 0.12
    xoffset = 0.0
    offset = 0.0
    vmin, vmax = 0.1 * vel_corr.min(), 1.02 * vel_corr.max()
    true_xy = (vmin + (0.44 + xoffset)*(vmax - vmin), (0.60+offset) * ymax)
    fv_xy = (vmin + (0.4 + xoffset)*(vmax-vmin), (0.52+offset)*ymax)
    logZ_eff_xy  = (vmin + (0.4 + xoffset)*(vmax-vmin), (0.44+offset)*ymax)

    fv_label = r'fv $= {:3.2f}$'.format(fv_data)
    logZ_eff_label = r'logZ_eff $= {:3.2f}$'.format(logZ_eff_data)

    axis.annotate('True', xy=true_xy, xytext=true_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25, style='italic')
    axis.annotate(fv_label, xy=fv_xy, xytext=fv_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(logZ_eff_label, xy=logZ_eff_xy, xytext=logZ_eff_xy, textcoords='data', xycoords='data', color='darkgreen', annotation_clip=False, fontsize=16, zorder=25)

    # error bar
    percent_lower = (1.0-0.6827)/2.0
    percent_upper = 1.0 - percent_lower
    param = np.median(samples_new, axis=0)
    param_lower = param - np.percentile(samples_new, 100*percent_lower, axis=0)
    param_upper = np.percentile(samples_new, 100*percent_upper, axis=0) - param

    infr_xy = (vmin + (0.74 + xoffset)*(vmax-vmin), (0.60+offset)*ymax)
    fv_xy  = (vmin + (0.685 + xoffset)*(vmax-vmin), (0.52+offset)*ymax)
    logZ_eff_xy = (vmin + (0.685 + xoffset) * (vmax - vmin), (0.44+offset) * ymax)

    fv_label = r'fv $= {:3.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[0], param_upper[0], param_lower[0])
    logZ_eff_label = r'logZ_eff $= {:3.2f}^{{+{:3.2f}}}_{{-{:3.2f}}}$'.format(param[1], param_upper[1], param_lower[1])

    axis.annotate('Inferred', xy=infr_xy, xytext=infr_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25, style='italic')
    axis.annotate(fv_label, xy=fv_xy, xytext=fv_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False,fontsize=16, zorder=25)
    axis.annotate(logZ_eff_label, xy=logZ_eff_xy, xytext=logZ_eff_xy, textcoords='data', xycoords='data', color='red', annotation_clip=False, fontsize=16, zorder=25)

    for ind in range(nrand):
        label = 'posterior draws' if ind == 0 else None
        axis.plot(vel_corr, factor*xi_model_rand[ind, :], linewidth=0.4, color='cornflowerblue', alpha=0.6, zorder=0, label=label)

    axis.tick_params(right=True, which='both')
    axis.minorticks_on()
    axis.set_xlim((vmin, vmax))
    axis.set_ylim((ymin, ymax))

    # Make the new upper x-axes in cMpc
    z = params['z'][0]
    cosmo = FlatLambdaCDM(H0=100.0 * params['lit_h'][0], Om0=params['Om0'][0], Ob0=params['Ob0'][0])
    Hz = (cosmo.H(z))
    a = 1.0 / (1.0 + z)
    rmin = (vmin * u.km / u.s / a / Hz).to('Mpc').value
    rmax = (vmax * u.km / u.s / a / Hz).to('Mpc').value
    atwin = axis.twiny()
    atwin.set_xlabel('R [cMpc]', fontsize=26, labelpad=8)
    atwin.xaxis.tick_top()
    # atwin.yaxis.tick_right()
    atwin.axis([rmin, rmax, ymin, ymax])
    atwin.tick_params(top=True)
    atwin.xaxis.set_minor_locator(AutoMinorLocator())
    atwin.tick_params(axis="x", labelsize=16)

    axis.annotate('CIV doublet', xy=(700, 0.90 * ymax), xytext=(700, 0.90* ymax), fontsize=16, color='black')
    axis.annotate('separation', xy=(710, 0.84 * ymax), xytext=(710, 0.84 * ymax), fontsize=16, color='black')
    axis.annotate('', xy=(520, 0.88 * ymax), xytext=(680, 0.88* ymax),
                fontsize=16, arrowprops={'arrowstyle': '-|>', 'lw': 4, 'color': 'black'}, va='center', color='black')

    vel_doublet = vel_metal_doublet('C IV', returnVerbose=False)
    axis.vlines(vel_doublet.value, ymin, ymax, color='k', linestyle='--', linewidth=2)

    # Plot a vertical line at the MgII doublet separation
    #vel_mg = vel_mgii()
    #axis.vlines(vel_mg.value, ymin, ymax, color='black', linestyle='--', linewidth=1.2)

    axis.legend(fontsize=16,loc='lower left', bbox_to_anchor=(1350, 0.69*ymax), bbox_transform=axis.transData)

    #plt.tight_layout()
    plt.savefig(outpath_local + 'fit_fv_logZ.pdf')
    #pdb.set_trace()

    plt.close()
