import sys

sys.path.insert(0, "/home/xinsheng/enigma/CIV_forest/") # inference_enrichment and halos_skewers dictionary
sys.path.insert(0, "/home/xinsheng/enigma/enigma/enigma/reion_forest/") # engima_reion_forest dictionary
sys.path.insert(0, "/home/xinsheng/enigma/code/") # CIV_lya_correlation.py dictionary

import inference_enrichment as ie
import pdb
from compute_model_grid import read_model_grid
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import CIV_lya_correlation as CIV_lya
from matplotlib import cm
import os
from astropy.table import Table
from astropy.io import fits
import pickle
import emcee
import neutral_center_colormap # colorbar from Lizhong Zhang, remove it if you don't need
import halos_skewers

modelfile = '/home/xinsheng/enigma/output/new_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_25.fits'

nproc = 30 # number of cores
k = 3 # just use in the name the file
logM_guess = 9.1
R_guess = 2.0
logZ_guess = -3.60
outpath_local = '/home/xinsheng/enigma/output/' # output path
linear_prior = False
cov = True
fvfm_file = '/home/xinsheng/enigma/fvfm/fvfm_all.fits' # path of fvfm_all.fits
overplot = False
# nlogM = 25 # grid of logM
# nR = 29 # grid of R_mpc
# nlogZ = 26 # grid of logZ

nlogM = 251 # grid of logM
nR = 291 # grid of R_mpc
nlogZ = 251 # grid of logZ
nwalker = 30 # number of walkers
walklength = 300000 # walklength in mcmc


logM_coarse = np.arange(8.5, 11.0+0.1, 0.1)
R_coarse = np.arange(0.1, 3.0+0.1, 0.1)
logZ_coarse = np.linspace(-4.5, -2.0, 26)

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = ie.read_model_grid(modelfile)

xi_model_fine, lndet_fine, covar_fine, logM_fine, R_fine, logZ_fine = CIV_lya.interp_covar(logM_coarse, R_coarse, logZ_coarse, xi_model_array, covar_array, lndet_array, nlogM, nR, nlogZ, nproc=30)
with open(outpath_local +'covar.npy', 'wb') as f:
    np.save(f,params)
    np.save(f,np.array(xi_model_fine))
    np.save(f,np.array(lndet_fine))
    np.save(f,np.array(covar_fine))
    np.save(f,np.array(logM_fine))
    np.save(f,np.array(R_fine))
    np.save(f,np.array(logZ_fine))
# plot_probability(init_out, logM_fine_cov, R_fine_cov, logZ_fine_cov, lnlike_fine_cov, output_local=outpath_local, fine=True, savefig='probability_cov.png')
# ie.plot_mcmc(sampler, param_samples, init_out, params, logM_fine_cov, R_fine_cov, logZ_fine_cov, xi_model_fine_cov, linear_prior, outpath_local, overplot=False, overplot_param=None, fvfm_file=fvfm_file)
