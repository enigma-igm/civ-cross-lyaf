"""
In this module:
    - imap_unordered_bar
    - mock_mean_covar #*
    - read_model_grid
    - compute_model_metal #*
    - get_npath
    - parser
    - main
"""
# Use single cores (forcing it for numpy operations)
import os
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from astropy.io import fits
import glob
from matplotlib import rcParams
import time
import sys
sys.path.insert(0, "/home/xinsheng/enigma/CIV_forest/")
sys.path.insert(0, "/home/xinsheng/enigma/enigma/enigma/reion_forest/")
sys.path.insert(0, "/home/xinsheng/enigma/code/")

import matplotlib.cm as cm
from tqdm.auto import tqdm

from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from astropy import constants as const
import scipy.ndimage
from astropy.table import Table, hstack, vstack
from IPython import embed
import utils
#import tqdm
#import enigma.reion_forest.istarmap  # import to apply patch
from multiprocessing import Pool
from tqdm import tqdm
import time
import CIV_lya_correlation as CIV_lya
import gc

def custom_cf_bin4(dv1=10, dv2=50, v_end=1000, v_min=10, v_max=2000):

    # linear around peak and small-scales
    v_bins1 = np.arange(v_min, v_end + dv1, dv1)
    v_lo1 = v_bins1[:-1]
    v_hi1 = v_bins1[1:]

    # larger linear bin size
    v_bins2 = np.arange(v_end, v_max + dv2, dv2)
    v_lo2 = v_bins2[:-1]
    v_hi2 = v_bins2[1:]

    v_lo_all = np.concatenate((v_lo1, v_lo2))
    v_hi_all = np.concatenate((v_hi1, v_hi2))

    return v_lo_all, v_hi_all

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

def mock_mean_covar(xi, xi_mean, npath, ncovar, nmock, seed=None):

    """
    Computes the covariance of the 2PCF from realizations of mock dataset.

    Args:
        xi (ndarray): 2PCF of *each* skewer; output of reion_forest.utils.compute_xi
        xi_mean (1D array): mean 2PCF averaging over the 2PCF of all skewers
        npath (int): number of skewers to use for mock data set; determined from nqsos and delta_z
        ncovar (int): number of mock realizations to generate
        nmock (int): number of mock realizations to save/output
        seed: if None, then randomly generate a seed.

    Returns:
        xi_mock_keep (2D array)
        covar (2D array)
    """

    rand = np.random.RandomState(seed) if seed is None else seed
    nskew, ncorr = xi.shape
    # Compute the mean from the "perfect" models
    xi_mock_keep = np.zeros((nmock, ncorr))
    covar = np.zeros((ncorr, ncorr))
    indx = np.arange(nskew)

    for imock in range(ncovar):
        ranindx = rand.choice(indx, replace=False, size=npath) # return random sampling of 'indx' of size 'npath'
        xi_mock = np.mean(xi[ranindx, :], axis=0) # mean 2PCF averaging over npath skewers;  for the mock dataset
        delta_xi = xi_mock - xi_mean
        covar += np.outer(delta_xi, delta_xi)
        if imock < nmock: # saving the first nmock results
            xi_mock_keep[imock, :] = xi_mock

    # Divide by ncovar since we estimated the mean from "independent" data; Eqn (13)
    covar /= ncovar

    return xi_mock_keep, covar

def read_model_grid(modelfile):

    """
    Extract saved outputs from running this module.
    """

    hdu = fits.open(modelfile)
    param = Table(hdu[1].data)
    xi_mock_array = hdu[2].data
    xi_model_array = hdu[3].data
    covar_array = hdu[4].data
    icovar_array = hdu[5].data
    lndet_array = hdu[6].data

    return param, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array

def compute_model_metal_lya(args):

    # unpacking input args
    i_R, i_logM, iZ, Rval, logMval, logZ, seed, taupath, fwhm, sampling, SNR, vmin_corr, vmax_corr, dv_corr1, dv_corr2, v_end, npath, ncovar, nmock, metal_ion = args

    if dv_corr1 < fwhm/sampling:
        raise Exception("dv_corr has to be greater or equal to fwhm/sampling=%0.1f" % (fwhm/sampling))

    # tau files for creating metal forest and computing CF
    rantaufile = os.path.join(taupath + 'rand_skewers_z45_ovt_xciv_tau_R_{:4.2f}'.format(Rval) + '_logM_{:4.2f}'.format(logMval) + '.fits') # 10,000 skewers

    lya_file = '/home/xinsheng/enigma/lya_forest/rand_skewers_z45_ovt_tau.fits'

    rand = np.random.RandomState(seed)
    params = Table.read(rantaufile, hdu=1)
    skewers = Table.read(rantaufile, hdu=2)
    #skewers = skewers[0:100] # testing

    # Generate skewers for rantaufile. This takes 5.36s for 10,000 skewers
    print("Computing the forest skewers ...")
    vel_lores, flux_lores = CIV_lya.create_metal_forest_short(params, skewers, logZ, fwhm, metal_ion=metal_ion, sampling=sampling, cgm_dict=None)

    del skewers

    gc.collect()

    params_lya = Table.read(lya_file, hdu=1)
    skewers_lya = Table.read(lya_file, hdu=2)

    vel_lores_lya, flux_lores_lya = CIV_lya.create_lya_forest_short(params_lya, skewers_lya, logZ, fwhm, metal_ion=metal_ion, sampling=sampling, cgm_dict=None)

    del skewers_lya

    gc.collect()


    # Add noise to the total flux
    print("Adding random noise to forest skewers ...")
    noise = rand.normal(0.0, 1.0 / SNR, flux_lores.shape)
    flux_noise = flux_lores + noise

    noise_lya = rand.normal(0.0, 1.0 / SNR, flux_lores_lya.shape)
    flux_noise_lya = flux_lores_lya + noise_lya

    # Compute delta_f
    print("Generating delta fields ...")
    mean_flux = np.mean(flux_noise) # flux + random noise
    delta_f = (flux_noise - mean_flux) / mean_flux
    mean_flux_nless = np.mean(flux_lores) # flux only
    delta_f_nless = (flux_lores - mean_flux_nless) / mean_flux_nless

    mean_flux_lya = np.mean(flux_noise_lya) # flux + random noise
    delta_f_lya = (flux_noise_lya - mean_flux_lya) / mean_flux_lya
    mean_flux_nless_lya = np.mean(flux_lores_lya) # flux only
    delta_f_nless_lya = (flux_lores_lya - mean_flux_nless_lya) / mean_flux_nless_lya

    # Compute xi (with noise). This takes 46.9s for 10,000 skewers
    print("Computing the 2PCF for noisy and noiseless data...")
    start_cf = time.time()
    (vel_mid, xi, npix, xi_zero_lag) = CIV_lya.compute_xi_CIV_lya_double_bin(delta_f, delta_f_lya, vel_lores, vel_lores_lya, vmin_corr, vmax_corr, dv_corr1, dv_corr2, v_end)
    end_cf = time.time()
    print("          done in %0.1f min" % ((end_cf - start_cf) / 60.))

    # Compute xi_noiseless, and use this to compute the mean.
    start_cf = time.time()
    (vel_mid, xi_nless, npix, xi_nless_zero_lag) = CIV_lya.compute_xi_CIV_lya_double_bin(delta_f_nless, delta_f_nless_lya, vel_lores, vel_lores_lya, vmin_corr, vmax_corr, dv_corr1, dv_corr2, v_end)
    xi_mean = np.mean(xi_nless, axis=0)
    end_cf = time.time()
    print("          done in %0.1f min" % ((end_cf - start_cf) / 60.))

    # Compute the covariance 569 ms for 10,000 skewers
    print("Computing the covariance matrix ...")
    start_cov = time.time()
    xi_mock, covar = mock_mean_covar(xi, xi_mean, npath, ncovar, nmock, seed=rand)
    icovar = np.linalg.inv(covar)  # invert the covariance matrix # need to be changed #######
    sign, logdet = np.linalg.slogdet(covar)  # compute the sign and natural log of the determinant of the covariance matrix
    end_cov = time.time()
    print("          done in %0.1f min" % ((end_cov - start_cov) / 60.))

    return i_logM, i_R, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet

def get_npath(params, skewers, logZ, fwhm, sampling, nqsos, delta_z):

    """
    Convenience function to compute npath, if don't want to run through entire main() function.
    """
    z = params['z'][0]
    # Determine our path length by computing one model, since this depends on the padding skewer size, etc.
    c_light = (const.c.to('km/s')).value
    z_min = z - delta_z
    z_eff = (z + z_min) / 2.0
    dv_path = (z - z_min) / (1.0 + z_eff) * c_light
    vel_lores, flux_lores, vel_hires, flux_hires, (oden, v_los, T, x_metal), cgm_tup = \
        utils.create_metal_forest(params, skewers[0:1], logZ, fwhm, metal_ion='C IV', sampling=sampling, cgm_dict=None)

    vside_lores = vel_lores.max() - vel_lores.min()
    vside_hires = vel_hires.max() - vel_hires.min()
    dz_side = (vside_lores / c_light) * (1.0 + z_eff)
    npath_float = nqsos * dv_path / vside_lores
    npath = int(np.round(npath_float))
    dz_tot = dz_side * npath
    print('Requested path length for nqsos={:d}'.format(nqsos) + ' covering delta_z={:5.3f}'.format(delta_z) +
          ' corresponds to requested total dz_req = {:5.3f}'.format(delta_z * nqsos) + ',  or {:5.3f}'.format(
        npath_float) +
          ' total skewers. Rounding to {:d}'.format(npath) + ' or dz_tot={:5.3f}'.format(dz_tot) + '.')

    return npath

#######################################################
def parser():
    import argparse

    parser = argparse.ArgumentParser(description='Create random skewers for CIV forest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nproc', type=int, default=40, help="Number of processors to run on")
    parser.add_argument('--fwhm', type=float, default=10.0, help="spectral resolution in km/s")
    parser.add_argument('--samp', type=float, default=3.0, help="Spectral sampling: pixels per fwhm resolution element")
    parser.add_argument('--SNR', type=float, default=50.0, help="signal-to-noise ratio")
    parser.add_argument('--nqsos', type=int, default=25, help="number of qsos")
    parser.add_argument('--delta_z', type=float, default=0.8, help="redshift pathlength per qso")
    parser.add_argument('--vmin', type=float, default=10.0, help="Minimum of velocity bins for correlation function")
    parser.add_argument('--vmax', type=float, default=2000, help="Maximum of velocity bins for correlation function")
    parser.add_argument('--dv1', type=float, default=10.0, help="Width of velocity bins for correlation function for selected range. "
                                                               "If not set fwhm will be used")
    parser.add_argument('--dv2', type=float, default=50.0, help="Width of velocity bins for correlation function for unselected range. ")
    parser.add_argument('--v_end', type=float, default=1000.0, help="Divide the selected and unselected range.")
    parser.add_argument('--ncovar', type=int, default=1000000, help="number of mock datasets for computing covariance") # small number for test run
    parser.add_argument('--nmock', type=int, default=500, help="number of mock datasets to store") # mock dataset made up of npath skewers
    parser.add_argument('--seed', type=int, default=1259761, help="seed for random number generator")
    parser.add_argument('--nlogZ', type=int, default=14, help="number of bins for logZ models")
    parser.add_argument('--logZmin', type=float, default=-4.5, help="minimum logZ value")
    parser.add_argument('--logZmax', type=float, default=-2.0, help="maximum logZ value")
    parser.add_argument('--logM_interval', type=float, default=0.2, help="The interval for each logM step")
    parser.add_argument('--R_Mpc_interval', type=float, default=0.2, help="The interval for each R step")

    return parser.parse_args()

def main():

    # reading in command line arguments
    args = parser()
    nproc = args.nproc
    fwhm = args.fwhm
    sampling = args.samp
    SNR = args.SNR
    nqsos = args.nqsos
    delta_z = args.delta_z
    ncovar = args.ncovar
    nmock = args.nmock
    seed = args.seed
    vmin_corr = args.vmin
    vmax_corr = args.vmax
    dv_corr1 = args.dv1 if args.dv1 is not None else fwhm # set dv_corr = fwhm if not provided
    dv_corr2 = args.dv2
    v_end = args.v_end
    metal_ion = 'C IV' # for now, hardcoding this
    logM_in = args.logM_interval
    R_in = args.R_Mpc_interval

    # Grid of metallicities
    nlogZ = args.nlogZ
    logZ_min = args.logZmin
    logZ_max = args.logZmax
    logZ_vec = np.linspace(logZ_min, logZ_max, nlogZ)

    # Grid of enrichment models
    logM = np.arange(8.5, 11.0+0.1, logM_in)
    R = np.arange(0.1, 3.0+0.1, R_in)
    nlogM, nR = len(logM), len(R)

    ### For testing ###
    # logZ_vec = logZ_vec[0:2]
    # logM = logM[0:3]
    # R = R[0:2]
    # nlogM, nR = len(logM), len(R)

    outpath = '/home/xinsheng/enigma/output/'
    outfile = outpath + 'test_corr_func_models_fwhm_{:5.3f}_samp_{:5.3f}_SNR_{:5.3f}_nqsos_{:d}'.format(fwhm, sampling, SNR, nqsos) + '.fits'

    taupath = '/home/xinsheng/enigma/tau/'
    taufiles = glob.glob(os.path.join(taupath, '*.fits'))

    # these files only for determining the path length
    params = Table.read(taufiles[0],hdu=1)
    skewers = Table.read(taufiles[0], hdu=2)

    # Determine our path length by computing one model, since this depends on the padding skewer size, etc.
    z = params['z'][0]
    c_light = (const.c.to('km/s')).value
    z_min = z - delta_z
    z_eff = (z + z_min) / 2.0
    dv_path = (z - z_min) / (1.0 + z_eff) * c_light

    vel_lores, flux_lores, vel_hires, flux_hires,  (oden, v_los, T, x_metal), cgm_tup = \
        utils.create_metal_forest(params, skewers[0:1], logZ_vec[0], fwhm, metal_ion=metal_ion, sampling=sampling, z=z, cgm_dict=None)

    vside_lores = vel_lores.max() - vel_lores.min()
    vside_hires = vel_hires.max() - vel_hires.min()
    dz_side = (vside_lores / c_light) * (1.0 + z_eff)
    npath_float = nqsos * dv_path / vside_lores
    npath = int(np.round(npath_float))
    dz_tot = dz_side*npath
    print('Requested path length for nqsos={:d}'.format(nqsos) + ' covering delta_z={:5.3f}'.format(delta_z) +
    ' corresponds to requested total dz_req = {:5.3f}'.format(delta_z * nqsos) + ',  or {:5.3f}'.format(npath_float) +
    ' total skewers. Rounding to {:d}'.format(npath) + ' or dz_tot={:5.3f}'.format(dz_tot) + '.')
    # Number of mock observations to create
    #pbar = tqdm(total=nlogZ*nhi, desc="Computing models")

    # Create the iterable argument list for map
    #args = (xhi_path, zstr, fwhm, sampling, SNR, vmin_corr, vmax_corr, dv_corr, npath, ncovar, nmock)
    args = (taupath, fwhm, sampling, SNR, vmin_corr, vmax_corr, dv_corr1, dv_corr2, v_end, npath, ncovar, nmock, metal_ion)
    all_args = []

    # giving the same random seed to all models, which is what we want to reduce random stochasticity and
    # to ensure models deform continuously with model parameters
    seed_vec = np.full(nlogM * nR * nlogZ, seed)

    for i_R, Rval in enumerate(R):
        for i_logM, logMval in enumerate(logM):
            for iZ, logZ in enumerate(logZ_vec):
                itup = (i_R, i_logM, iZ, Rval, logMval, logZ, seed_vec[i_R]) + args # doesn't matter what index used for seed_vec since same seed is being used
                all_args.append(itup)

    print('Computing nmodel={:d} models on nproc={:d} processors'.format(nlogM * nR * nlogZ, nproc))

    output = imap_unordered_bar(compute_model_metal_lya, all_args, nproc)
    #pool = Pool(processes=nproc)
    #output = pool.starmap(compute_model, all_args)

    # Initialize the arrays to hold everything
    ilogM, iR, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet = output[0]
    ncorr = vel_mid.shape[0]
    xi_mock_array = np.zeros((nlogM, nR, nlogZ,) + xi_mock.shape)
    xi_mean_array = np.zeros((nlogM, nR, nlogZ,) + xi_mean.shape)
    covar_array = np.zeros((nlogM, nR, nlogZ,) + covar.shape)
    icovar_array = np.zeros((nlogM, nR, nlogZ,) + icovar.shape)
    lndet_array = np.zeros((nlogM, nR, nlogZ))

    # Unpack the output
    for out in output:
        ilogM, iR, iZ, vel_mid, xi_mock, xi_mean, covar, icovar, logdet = out
        # Right out a random subset of these so that we don't use so much disk.
        xi_mock_array[ilogM, iR, iZ, :, :] = xi_mock
        xi_mean_array[ilogM, iR, iZ, :] = xi_mean
        covar_array[ilogM, iR, iZ, :, :] = covar
        icovar_array[ilogM, iR, iZ, :, :] = icovar
        lndet_array[ilogM, iR, iZ] = logdet

    nskew = len(skewers)
    param_model=Table([[nqsos], [delta_z], [dz_tot], [npath], [ncovar], [nmock], [fwhm], [sampling], [SNR], [nskew], [seed], \
                       [nlogM], [logM], [nR], [R], [nlogZ], [logZ_vec], [ncorr], [vmin_corr],[vmax_corr], [vel_mid]], \
                      names=('nqsos', 'delta_z', 'dz_tot', 'npath', 'ncovar', 'nmock', 'fwhm', 'sampling', 'SNR', 'nskew', 'seed', \
                             'nlogM', 'logM', 'nR', 'R_Mpc', 'nlogZ', 'logZ', 'ncorr', 'vmin_corr', 'vmax_corr', 'vel_mid'))
    param_out = hstack((params, param_model))
    # Write out to multi-extension fits
    print('Writing out to disk')
    # Write to outfile
    hdu_param = fits.table_to_hdu(param_out)
    hdu_param.name = 'METADATA'
    hdulist = fits.HDUList()
    hdulist.append(hdu_param)
    hdulist.append(fits.ImageHDU(data=xi_mock_array, name='XI_MOCK')) # XI_MOCK_ARRAY
    hdulist.append(fits.ImageHDU(data=xi_mean_array, name='XI_MEAN'))
    hdulist.append(fits.ImageHDU(data=covar_array, name='COVAR'))
    hdulist.append(fits.ImageHDU(data=icovar_array, name='ICOVAR'))
    hdulist.append(fits.ImageHDU(data=lndet_array, name='LNDET'))
    hdulist.writeto(outfile, overwrite=True)


if __name__ == '__main__':
    main()
