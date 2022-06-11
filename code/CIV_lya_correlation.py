#### function for CIV and lya cross-correlation
#### I move all of my modified functions here in order to better organize them

import sys
'''
xi_sum_CIV_lya
compute_xi_all_CIV_lya
compute_xi_CIV_lya
create_lya_forest
create_metal_forest_tau
create_metal_forest_red
create_metal_forest_blue
'''

sys.path.insert(0, "/Users/xinsheng/CIV_forest/")
sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")

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
from enigma.reion_forest.compute_model_grid_civ import read_model_grid
import halos_skewers

from enigma.reion_forest.utils import *
from metal_corrfunc import *

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
                   cgm_dict=None, metal_dndz_func=None, cgm_seed=None, want_hires=True):

    # similar as enigma.reion_forest.fig_corrfunc.py
    # if sampling not provided, then default to sampling=3


    # vel_lores_CIV, (flux_lores_tot_CIV, flux_lores_igm_CIV, flux_lores_cgm_CIV), \
    # vel_hires_CIV, (flux_hires_tot_CIV, flux_hires_igm_CIV, flux_hires_cgm_CIV), \
    # (oden_CIV, v_los_CIV, T_CIV, x_metal_CIV), cgm_tup_CIV, tau_igm_CIV = create_metal_forest(params_CIV, skewers_CIV, logZ, fwhm, metal_ion, sampling=sampling, \
    #                                                                      cgm_dict=cgm_dict, metal_dndz_func=metal_dndz_func, seed=cgm_seed)
    #
    # vel_lores_lya, (flux_lores_tot_lya, flux_lores_igm_lya, flux_lores_cgm_lya), \
    # vel_hires_lya, (flux_hires_tot_lya, flux_hires_igm_lya, flux_hires_cgm_lya), \
    # (oden_lya, T_lya, x_metal_lya), cgm_tup_lya, tau_igm_lya = create_lya_forest(params_lya, skewers_lya, fwhm, sampling=sampling)

# test only

    vel_lores_CIV, (flux_lores_tot_CIV, flux_lores_igm_CIV, flux_lores_cgm_CIV), \
    vel_hires_CIV, (flux_hires_tot_CIV, flux_hires_igm_CIV, flux_hires_cgm_CIV), \
    (oden_CIV, v_los_CIV, T_lya,x_metal_CIV), cgm_tup_CIV = reion_utils.create_metal_forest(params_CIV, skewers_CIV, logZ, fwhm, metal_ion, sampling=sampling, \
                                                                          cgm_dict=cgm_dict, metal_dndz_func=metal_dndz_func, seed=cgm_seed)

    vel_lores_lya, (flux_lores_tot_lya, flux_lores_igm_lya, flux_lores_cgm_lya), \
    vel_hires_lya, (flux_hires_tot_lya, flux_hires_igm_lya, flux_hires_cgm_lya), \
    (oden_lya, v_los_lya, T_lya, x_metal_lya),cgm_tup_CIV = reion_utils.create_metal_forest(params_CIV, skewers_CIV, logZ, fwhm, metal_ion, sampling=sampling, \
                                                                          cgm_dict=cgm_dict, metal_dndz_func=metal_dndz_func, seed=cgm_seed)

    flux_hires_tot_lya = flux_lores_tot_lya
    vel_hires_lya = vel_lores_lya
    flux_hires_tot_CIV = flux_lores_tot_CIV
    vel_hires_CIV = vel_lores_CIV
    # Compute mean flux and delta_flux
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


def compute_xi_CIV_lya(delta_f_in_CIV, delta_f_in_lya, vel_spec_CIV, vel_spec_lya, vmin, vmax, dv, gpm=None, progress=False):
    """

    Args:
        delta_f_in (float ndarray), shape (nskew, nspec) or (nspec,):
            Flux contrast array
        vel_spec (float ndarray): shape (nspec,)
            Velocities for flux contrast
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


def create_lya_forest(params, skewers, fwhm, z=None, sampling=3.0, cgm_dict=None, seed=None):
    """
        Generate lya line forest, with the option to include CGM absorbers.

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
    #tau0, nh_bar = lya_tau0(z, cosmo=Planck15, X=0.76)
    # note that tau0 is obtained from the stronger blue line

    # Pad the skewer for the convolution
    npad = int(np.ceil((7.0*fwhm)/dvpix_hires)) ## v_metal = 0 and f_ratio = 1 is possible?
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
    tau_interp = interp1d(vel_pad, tau_igm, axis=1,fill_value=0.0, kind='cubic', bounds_error=False)
    tau_plot = tau_igm[:,iobs_hires]
    # tau_igm: total tau is the sum of the red and blue tau's

    # Now generate the CGM
    # if cgm_dict != None:
    #     # tau_cgm, logN_draws, b_draws, v_draws, W_2796_draws, iskew_abs, tau_draws = create_mgii_cgm(vel_pad, nskew, z, cgm_dict, rand=rand)
    #     tau_cgm, logN_draws, b_draws, v_draws, W_blue_draws, iskew_abs, tau_draws = create_metal_cgm(vel_pad, nskew, z, cgm_dict, metal_dndz_func, metal_ion=metal_ion, seed=seed)
    #
    #     # Only pass back the draws that reside in the final velocity coverage (as determined by vel_lores)
    #     ikeep = (v_draws > vel_lores.min()) & (v_draws < vel_lores.max())
    #     cgm_tuple = (logN_draws[ikeep], b_draws[ikeep], v_draws[ikeep], W_blue_draws[ikeep], iskew_abs[ikeep], tau_draws[ikeep, :])
    # else:
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


def create_metal_forest_tau(params, skewers, logZ, fwhm, metal_ion, z=None, sampling=3.0, cgm_dict=None, metal_dndz_func=None, seed=None):
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


def create_metal_forest_red(params, skewers, logZ, fwhm, metal_ion, z=None, sampling=3.0, cgm_dict=None, metal_dndz_func=None, seed=None):
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
    tau_igm = tau_red # total tau is the sum of the red and blue tau's

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



def create_metal_forest_blue(params, skewers, logZ, fwhm, metal_ion, z=None, sampling=3.0, cgm_dict=None, metal_dndz_func=None, seed=None):
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
    tau_igm = tau_blue # total tau is the sum of the red and blue tau's

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
