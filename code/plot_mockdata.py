import sys
sys.path.insert(0, "/Users/xinsheng/CIV_forest/")
sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")
sys.path.insert(0,"/Users/xinsheng/civ-cross-lyaf/code")

import inference_enrichment as ie
import pdb
from enigma.reion_forest.compute_model_grid import read_model_grid

modelfile = '/Users/xinsheng/civ-cross-lyaf/enrichment_models/corrfunc_models/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
logM_guess = 9.2
R_guess = 1.3
logZ_guess = -3.5

# logM_guess = 9.5
# R_guess = 0.9
# logZ_guess = -3.25

linear_prior = False

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = ie.read_model_grid(modelfile)

init_out = ie.init(modelfile, logM_guess, R_guess, logZ_guess)

lnlike_coarse, lnlike_fine, xi_model_fine, logM_fine, R_fine, logZ_fine = ie.interp_likelihood(init_out, 3, 4, 5, interp_lnlike=True, interp_ximodel=True)

sampler, param_samples, bounds = ie.mcmc_inference(10000, 1000, 100, logM_fine, R_fine, logZ_fine, lnlike_fine, linear_prior)

ie.plot_mcmc(sampler, param_samples, init_out, params, logM_fine, R_fine, logZ_fine, xi_model_fine, linear_prior)
