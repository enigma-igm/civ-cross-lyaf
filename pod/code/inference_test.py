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
test_num = 1000
k = 3 # just use in the name the file
logM_guess = 9.9
R_guess = 1.0
logZ_guess = -3.60
outpath_local = '/home/xinsheng/enigma/output/inference_test/' # output path
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

if ( os.path.isfile(outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.out' % (logM_guess, R_guess, logZ_guess, k)) == False ):
    init_out = ie.init(modelfile, logM_guess, R_guess, logZ_guess)
    pickle.dump(init_out, open(outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.out' % (logM_guess, R_guess, logZ_guess, k), 'wb'))

else:
    init_out = pickle.load(open(outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.out' % (logM_guess, R_guess, logZ_guess, k), 'rb'))

if ( os.path.isfile(outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.npy' % (logM_guess, R_guess, logZ_guess, k)) == False ):
    lnlike_fine_cov, xi_model_fine_cov, logM_fine_cov, R_fine_cov, logZ_fine_cov = CIV_lya.interp_likelihood_nproc(init_out, nlogM, nR, nlogZ, covar_path, nproc=nproc)
    with open(outpath_local +'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.npy' % (logM_guess, R_guess, logZ_guess, k), 'wb') as f:
        np.save(f,k)
        np.save(f,params)
        np.save(f,np.array(lnlike_fine_cov))
        np.save(f,np.array(xi_model_fine_cov))
        np.save(f,np.array(logM_fine_cov))
        np.save(f,np.array(R_fine_cov))
        np.save(f,np.array(logZ_fine_cov))

else:
    with open(outpath_local+'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.npy' % (logM_guess, R_guess, logZ_guess, k), 'rb') as f:
        k_test = np.load(f)
        params = np.load(f)
        lnlike_fine_cov = np.load(f)
        xi_model_fine_cov = np.load(f)
        logM_fine_cov = np.load(f)
        R_fine_cov = np.load(f)
        logZ_fine_cov = np.load(f)

# logM_mid, R_mid, logZ_mid = CIV_lya.inference_test_CIV_lya(walklength, 1000, nwalker, logM_fine_cov, R_fine_cov, logZ_fine_cov, lnlike_fine_cov, linear_prior, nproc = nproc, test_num = test_num)
#
# with open(outpath_local +'save_mid_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.npy' % (logM_guess, R_guess, logZ_guess, k), 'wb') as f:
#
#     np.save(f,np.array(logM_mid))
#     np.save(f,np.array(R_mid))
#     np.save(f,np.array(logZ_mid))
