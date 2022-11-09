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
fvfm_file = '/home/xinsheng/enigma/fvfm/fvfm_all.fits' # path of fvfm_all.fits
covar_path = '/home/xinsheng/enigma/covar/'
nlogM = 251 # grid of logM
nR = 291 # grid of R_mpc
nlogZ = 251 # grid of logZ

logZ_coarse = np.linspace(-4.5, -2.0, 26)
logM_coarse = np.arange(8.5, 11.0+0.1, 0.1)
R_coarse = np.arange(0.1, 3.0+0.1, 0.1)

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = ie.read_model_grid(modelfile)

init_out = logM_coarse, R_coarse, logZ_coarse, xi_model_array, covar_array, lndet_array

CIV_lya.covar_fine_generator(init_out, nlogM, nR, nlogZ, covar_path, nproc=nproc)
