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
from utils import find_closest

import os

# os.environ["OMP_NUM_THREADS"] = "1"

from astropy.table import Table
from astropy.io import fits
import pickle
import emcee
import halos_skewers

modelfile_cross = '/home/xinsheng/enigma/data/cross_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_25.fits'
modelfile_auto = '/home/xinsheng/enigma/data/auto_corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'

nproc = 100 # number of cores
test_size = 100
logM_guess = 9.9
R_guess = 1.0
logZ_guess = -3.60
outpath_local = '/home/xinsheng/enigma/output/combine/mcmc_99_10_36/inference_test/' # output path
linear_prior = False
cov = True
fvfm_file = '/home/xinsheng/enigma/data/fvfm/fvfm_all.fits' # path of fvfm_all.fits
covar_path_cross = '/home/xinsheng/enigma/data/covar_cross/'
covar_path_auto = '/home/xinsheng/enigma/data/covar_auto/'
overplot = False
nlogM = 251 # grid of logM
nR = 291 # grid of R_mpc
nlogZ = 251 # grid of logZ
nwalker = 30 # number of walkers
walklength = 150000 # walklength in mcmc

params_auto, xi_mock_array_auto, xi_model_array_auto, covar_array_auto, icovar_array_auto, lndet_array_auto = ie.read_model_grid(modelfile_auto)
params_cross, xi_mock_array_cross, xi_model_array_cross, covar_array_cross, icovar_array_cross, lndet_array_cross = ie.read_model_grid(modelfile_cross)

def true_location(param_samples):

    param = np.median(param_samples, axis=0)

    logM_infer = param[0]
    R_infer = param[1]
    logZ_infer = param[2]

    cov_matrix = np.cov(param_samples.T, bias=True)
    eigenvalue, eigenvector = np.linalg.eig(cov_matrix)
    std = np.std(param_samples, axis=0)

    logM_std = std[0]
    R_std = std[1]
    logZ_std = std[2]

    infer_vector = np.array([[logM_infer, R_infer, logZ_infer]])
    True_vector = np.array([[logM_guess, R_guess, logZ_guess]])

    True_distance = (True_vector - infer_vector) @ cov_matrix @ (True_vector - infer_vector).T

    count = 0
    percentile = 0

    for x in param_samples:
        count += 1
        location = np.array([x])
        distance = (location - infer_vector) @ cov_matrix @ (location - infer_vector).T
        if True_distance < distance:
            percentile += 1

    fall_in = percentile / count

    return fall_in


def likelihood_calculation(num):

    logZ_coarse = np.round(params_auto['logZ'][0], 2)
    logM_coarse = np.round(params_auto['logM'][0], 2)
    R_coarse = np.round(params_auto['R_Mpc'][0], 2)
    ilogZ = find_closest(logZ_coarse, logZ_guess)
    ilogM =  find_closest(logM_coarse, logM_guess)
    iR = find_closest(R_coarse, R_guess)
    logZ_data = logZ_coarse[ilogZ]
    logM_data = logM_coarse[ilogM]
    R_data = R_coarse[iR]

    xi_data_auto = xi_mock_array_auto[ilogM, iR, ilogZ, num, :].flatten()
    xi_mask_auto = np.ones_like(xi_data_auto, dtype=bool)  # in case you want to mask any xi value, otherwise all True
    xi_data_cross = xi_mock_array_cross[ilogM, iR, ilogZ, num, :].flatten()
    xi_mask_cross = np.ones_like(xi_data_cross, dtype=bool)  # in case you want to mask any xi value, otherwise all True

    init_out_auto = logM_data, R_data, logZ_data, xi_data_auto, xi_mask_auto, xi_model_array_auto
    init_out_cross = logM_data, R_data, logZ_data, xi_data_cross, xi_mask_cross, xi_model_array_cross

    lnlike_fine_auto, xi_model_fine_auto, logM_fine_auto, R_fine_auto, logZ_fine_auto = CIV_lya.interp_likelihood_inference(init_out_auto, nlogM, nR, nlogZ, covar_path_auto, nproc=nproc)
    lnlike_fine_cross, xi_model_fine_cross, logM_fine_cross, R_fine_cross, logZ_fine_cross = CIV_lya.interp_likelihood_inference(init_out_cross, nlogM, nR, nlogZ, covar_path_cross, nproc=nproc)
    lnlike_fine = lnlike_fine_cross + lnlike_fine_auto

    return lnlike_fine_cross, lnlike_fine_auto, lnlike_fine

def inference_nproc(args):

    lnlike_fine_cross, lnlike_fine_auto, lnlike_fine, num = args

    logZ_coarse = np.linspace(-4.5, -2.0, 26)
    logM_coarse = np.arange(8.5, 11.0+0.1, 0.1)
    R_coarse = np.arange(0.1, 3.0+0.1, 0.1)

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

    chain_output_auto = outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_auto_%d' % (logM_guess, R_guess, logZ_guess, num) + '.fits'
    chain_output_cross = outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_cross_%d' % (logM_guess, R_guess, logZ_guess, num) + '.fits'
    chain_output = outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_combine_%d' % (logM_guess, R_guess, logZ_guess, num) + '.fits'

    if ( os.path.isfile(outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_combine_%d' % (logM_guess, R_guess, logZ_guess, num) + '.fits') == False ):

        sampler_auto, param_samples_auto, bounds_auto = ie.mcmc_inference(walklength, 1000, nwalker, logM_fine, R_fine, logZ_fine, lnlike_fine_auto, linear_prior, savefits_chain = chain_output_auto)
        sampler_cross, param_samples_cross, bounds_cross = ie.mcmc_inference(walklength, 1000, nwalker, logM_fine, R_fine, logZ_fine, lnlike_fine_cross, linear_prior, savefits_chain = chain_output_cross)
        sampler, param_samples, bounds = ie.mcmc_inference(walklength, 1000, nwalker, logM_fine, R_fine, logZ_fine, lnlike_fine, linear_prior, savefits_chain = chain_output)

    else:

        hdu_auto = fits.open(chain_output_auto)
        param_samples_auto = hdu_auto[3].data
        hdu_cross = fits.open(chain_output_cross)
        param_samples_cross = hdu_cross[3].data
        hdu_combine = fits.open(chain_output)
        param_samples = hdu_combine[3].data

    percent = true_location(param_samples)
    percent_auto = true_location(param_samples_auto)
    percent_cross = true_location(param_samples_cross)

    return percent, percent_auto, percent_cross, num

all_args = []
output_cross, output_auto, output_combine = [], [], []

for i in range(test_size):
    if ( os.path.isfile(outpath_local +'inference_likelihood_logM_%.2f_R_%.2f_logZ_%.2f_%d.npy' % (logM_guess, R_guess, logZ_guess, i)) == False ):
        likelihood_cross, likelihood_auto, likelihood_combine = likelihood_calculation(i)
        with open(outpath_local +'inference_likelihood_logM_%.2f_R_%.2f_logZ_%.2f_num_%d.npy' % (logM_guess, R_guess, logZ_guess, i), 'wb') as f:
            np.save(f, likelihood_combine)
            np.save(f, likelihood_auto)
            np.save(f, likelihood_cross)
    else:
        with open(outpath_local +'inference_likelihood_logM_%.2f_R_%.2f_logZ_%.2f_%d.npy' % (logM_guess, R_guess, logZ_guess, i), 'rb') as f:
            likelihood_combine = np.load(f)
            likelihood_auto = np.load(f)
            likelihood_cross = np.load(f)
    itup = (likelihood_cross, likelihood_auto, likelihood_combine, i)
    all_args.append(itup)

output = CIV_lya.imap_unordered_bar(inference_nproc, all_args, nproc)
percent_total_auto = []
percent_total_cross = []
percent_total_combine = []
num_total = []

for out in output:
    percent, percent_auto, percent_cross, num = out
    percent_total_auto.append(percent_auto)
    percent_total_cross.append(percent_cross)
    percent_total_combine.append(percent)
    num_total.append(num)

with open(outpath_local +'inference_result_logM_%.2f_R_%.2f_logZ_%.2f.npy' % (logM_guess, R_guess, logZ_guess), 'wb') as f:
    np.save(f,percent_total_combine)
    np.save(f,percent_total_auto)
    np.save(f,percent_total_cross)
    np.save(f, num_total)
