import sys
sys.path.insert(0, "/Users/xinsheng/CIV_forest/")
sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")
sys.path.insert(0,"/Users/xinsheng/civ-cross-lyaf/code")

import inference_enrichment as ie
import pdb
from enigma.reion_forest.compute_model_grid import read_model_grid
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

modelfile = '/Users/xinsheng/civ-cross-lyaf/enrichment_models/corrfunc_models/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
# logM_guess = 9.5
# R_guess = 1.0
# logZ_guess = -3.5

logM_guess = 9.9
R_guess = 1.0
logZ_guess = -3.6
outpath_local = '/Users/xinsheng/civ-cross-lyaf/output/mcmc/mcmc_1/'

linear_prior = False

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = ie.read_model_grid(modelfile)

init_out = ie.init(modelfile, logM_guess, R_guess, logZ_guess)

lnlike_coarse, lnlike_fine, xi_model_fine, logM_fine, R_fine, logZ_fine = ie.interp_likelihood(init_out, 13, 15, 15, interp_lnlike=True, interp_ximodel=True)

#
# norm = np.exp(lnlike_fine-2500).sum()
#
# print(norm)
#
# dR = R_fine[1] - R_fine[0]
# dZ = logZ_fine[1] - logZ_fine[0]
#
# fig = plt.figure(figsize = (15,10))
#
# plt.title('Probability distribution for logM = %.2f, R = %.2f and logZ = %.2f' % (logM_guess, R_guess, logZ_guess),y=1.1)
#
# plt.axis('off')
#
# for i in range(len(logM_fine)):
#     fig.add_subplot(n,len(logM_fine)/n,i+1)
#     plt.title('logM = %.1f' % logM_fine[i])
#     plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#     plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#     im = plt.imshow(np.exp(lnlike_fine[i]-2500)/norm, vmin=0, vmax=1, extent=[logZ_fine[0]-dZ/2, logZ_fine[-1]+dZ/2, R_fine[0]-dR/2, R_fine[-1]+dR/2])
#     for h in R_fine:
#         plt.axhline(h)
#     for v in logZ_fine:
#         plt.axvline(v)
#     plt.xticks(logZ_fine)
#     plt.yticks(R_fine)
#     plt.xlabel('logZ')
#     plt.ylabel('R_Mpc')
#
# plt.savefig('/Users/xinsheng/civ-cross-lyaf/output/mcmc/probability.png', layout = 'tight')
# plt.close()

sampler, param_samples, bounds = ie.mcmc_inference(15000, 1000, 60, logM_fine, R_fine, logZ_fine, lnlike_fine, linear_prior)
#
ie.plot_mcmc(sampler, param_samples, init_out, params, logM_fine, R_fine, logZ_fine, xi_model_fine, linear_prior, outpath_local)
