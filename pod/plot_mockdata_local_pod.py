import sys
sys.path.insert(0, "/Users/xinsheng/CIV_forest/")
sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")
sys.path.insert(0, "/Users/xinsheng/civ-cross-lyaf/code/")

import inference_enrichment as ie
import pdb
from enigma.reion_forest.compute_model_grid import read_model_grid
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

os.chdir("/Users/xinsheng/Athena_Radiation-master")
import neutral_center_colormap

def plot_probability(init_out, logM_fine, R_fine, logZ_fine, lnlike_fine, output_local, fine=False, savefig='probability.png'):
    logM_coarse, R_coarse, logZ_coarse, logM_data, R_data, logZ_data, xi_data, xi_mask, xi_model_array, \
    covar_array, icovar_array, lndet_array, vel_corr, logM_guess, R_guess, logZ_guess = init_out

    if fine == False:
        logM_fine = logM_coarse
        R_fine = R_coarse
        logZ_fine = logZ_coarse
        lnlike_fine = lnlike_coarse

    dR = R_fine[1] - R_fine[0]
    dZ = logZ_fine[1] - logZ_fine[0]
    dM = logM_fine[1] - logM_fine[0]

    fig = plt.figure(figsize = (15,10))

    logM_max, R_max, logZ_max = np.where(lnlike_fine==lnlike_fine.max())

    lnlike_fine_max = lnlike_fine[:,:,logZ_max].max()

    plt.title('logZ = %.2f' % logZ_fine[logZ_max])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    im = plt.imshow(lnlike_fine[:,:,logZ_max], vmin = lnlike_fine_max-10, vmax = lnlike_fine_max, origin = 'lower', \
    extent=[R_fine[0]-dR/2, R_fine[-1]+dR/2, logM_fine[0]-dM/2, logM_fine[-1]+dM/2],\
    cmap = 'NCcmap')
    print(logM_max, R_max, logZ_max)

    # for h in R_fine:
    #     plt.axhline(h)
    # for v in logM_fine:
    #     plt.axvline(v)

    logM_range = np.arange(logM_fine.min(),logM_fine.max(),0.2)
    R_range = np.arange(R_fine.min(),R_fine.max(),0.2)

    plt.yticks(logM_range)
    plt.xticks(R_range)
    plt.ylabel('logM')
    plt.xlabel('R_Mpc')

    plt.colorbar(im, label = 'lnL')
    plt.savefig(output_local + savefig)
    plt.close()

    #
    # dR = R_coarse[1] - R_coarse[0]
    # dZ = logZ_coarse[1] - logZ_coarse[0]
    # dM = logM_coarse[1] - logM_coarse[0]
    #
    # X = R_coarse
    # Y = logM_coarse
    #
    # X, Y = np.meshgrid(X, Y)
    #
    # fig = plt.figure(figsize = (15,10))
    #
    # plt.title('Probability distribution for logM = %.2f, R = %.2f and logZ = %.2f' % (logM_guess, R_guess, logZ_guess), y=1.1)
    #
    # logM_max, R_max, logZ_max = np.where(lnlike_coarse==lnlike_coarse.max())
    # Z = np.reshape(lnlike_coarse[:,:,logZ_max], (logM_coarse.size,R_coarse.size))
    # print(Z.shape)
    # print(X.shape,Y.shape)
    #
    # fig = plt.figure(figsize=(10,10))
    # ax = plt.axes(projection='3d')
    # plt.title('logZ = %.2f' % logZ_coarse[logZ_max])
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    # im = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    #
    # # plt.xticks(logM_coarse)
    # # plt.yticks(R_coarse)
    # plt.ylabel('logM')
    # plt.xlabel('R_Mpc')
    #
    # plt.colorbar(im)
    # plt.savefig('/Users/xinsheng/civ-cross-lyaf/output/mcmc/mcmc_1/probability.png')
    # plt.close()


modelfile = '/Users/xinsheng/civ-cross-lyaf/enrichment_models/corrfunc_models/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_25.fits'
#modelfile = '/Users/xinsheng/civ-cross-lyaf/enrichment_models/corrfunc_models/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_25.fits'
sstie_file = '/Users/xinsheng/civ-cross-lyaf/enrichment_models/corrfunc_models/mcmc_chain_Fig13.fits'

if sstie_file != None:
    hdu = fits.open(sstie_file)
    param_sstie = hdu[3].data
    print(hdu[3].header['EXTNAME'])

k = 2

logM_guess = 9.20
R_guess = 2.45
logZ_guess = -3.50
outpath_local = '/Users/xinsheng/civ-cross-lyaf/pod/output/mcmc_1/'
linear_prior = False

cov = True

print(os.path.isfile(outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.npy' % (logM_guess, R_guess, logZ_guess, k)))


if ( os.path.isfile(outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.npy' % (logM_guess, R_guess, logZ_guess, k)) == True ):

    sampler_name = outpath_local + 'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.h5' % (logM_guess, R_guess, logZ_guess, k)

    sampler = emcee.backends.HDFBackend(sampler_name)

    init_out = pickle.load(open(outpath_local+'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.out' % (logM_guess, R_guess, logZ_guess, k), 'rb'))

    with open(outpath_local+'save_logM_%.2f_R_%.2f_logZ_%.2f_k_%d.npy' % (logM_guess, R_guess, logZ_guess, k), 'rb') as f:

        k_test = np.load(f)

        params = np.load(f)

        lnlike_coarse_cov = np.load(f)
        lnlike_fine_cov = np.load(f)
        xi_model_fine_cov = np.load(f)
        logM_fine_cov = np.load(f)
        R_fine_cov = np.load(f)
        logZ_fine_cov = np.load(f)

        param_samples = np.load(f)

        bounds = np.load(f)

    plot_probability(init_out, logM_fine_cov, R_fine_cov, logZ_fine_cov, lnlike_fine_cov, output_local=outpath_local, fine=True, savefig='probability_cov.png')

    ie.plot_mcmc(sampler, param_samples, init_out, params, logM_fine_cov, R_fine_cov, logZ_fine_cov, xi_model_fine_cov, linear_prior, outpath_local, overplot=False, overplot_param=None)
