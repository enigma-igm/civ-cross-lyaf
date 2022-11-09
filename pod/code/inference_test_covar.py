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

os.environ["OMP_NUM_THREADS"] = "1"

from astropy.table import Table
from astropy.io import fits
import pickle
import emcee
import neutral_center_colormap # colorbar from Lizhong Zhang, remove it if you don't need
import halos_skewers


modelfile = '/home/xinsheng/enigma/output/new_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_25.fits'

nproc = 50 # number of cores
test_num = 100
k = 3 # just use in the name the file
logM_guess = 9.9
R_guess = 1.0
logZ_guess = -3.60
outpath_local = '/home/xinsheng/enigma/output/inference_test_covar/' # output path
linear_prior = False
fvfm_file = '/home/xinsheng/enigma/fvfm/fvfm_all.fits' # path of fvfm_all.fits
covar_path = '/home/xinsheng/enigma/covar/'
# nlogM = 25 # grid of logM
# nR = 29 # grid of R_mpc
# nlogZ = 26 # grid of logZ

nlogM = 251 # grid of logM
nR = 291 # grid of R_mpc
nlogZ = 251 # grid of logZ
nwalker = 30 # number of walkers
walklength = 150000 # walklength in mcmc

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = ie.read_model_grid(modelfile)
lnlike_fine = []
logM_tot = []
R_tot = []
logZ_tot = []
logZ_coarse = np.linspace(-4.5, -2.0, 26)
logM_coarse = np.arange(8.5, 11.0+0.1, 0.1)
R_coarse = np.arange(0.1, 3.0+0.1, 0.1)
logZ_data = logZ_coarse[ilogZ]
logM_data = logM_coarse[ilogM]
R_data = R_coarse[iR]
ilogZ = find_closest(logZ_coarse, logZ_guess)
ilogM =  find_closest(logM_coarse, logM_guess)
iR = find_closest(R_coarse, R_guess)

for i in range(100):
    xi_data = xi_mock_array[ilogM, iR, ilogZ, i, :].flatten()
    init_out = logM_data, R_data, logZ_data, xi_data, xi_mask
    lnlike_fine_cov, logM_fine_cov, R_fine_cov, logZ_fine_cov = CIV_lya.interp_likelihood_inference_test(init_out, nlogM, nR, nlogZ, covar_path, nproc=nproc)
    lnlike_fine.append(lnlike_fine_cov)

for i in range(10):
    logM_mid, R_mid, logZ_mid = CIV_lya.inference_test_CIV_lya(walklength, 1000, nwalker, logM_fine_cov, R_fine_cov, logZ_fine_cov, lnlike_fine, linear_prior, nproc = nproc, test_num = test_num)
    logM_tot.append(logM_mid)
    R_tot.append(R_mid)
    logZ_tot.append(logZ_mid)

with open(outpath_local +'save_mid_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.npy' % (logM_guess, R_guess, logZ_guess, k), 'wb') as f:

    np.save(f,np.array(logM_tot))
    np.save(f,np.array(R_tot))
    np.save(f,np.array(logZ_tot))
