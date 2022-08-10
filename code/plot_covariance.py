import sys
sys.path.insert(0, "/Users/xinsheng/CIV_forest/")
sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")
sys.path.insert(0,"/Users/xinsheng/civ-cross-lyaf/code")

import inference_enrichment as ie
import pdb
from enigma.reion_forest.compute_model_grid import read_model_grid
import numpy as np
import matplotlib.pyplot as plt

modelfile = '/Users/xinsheng/civ-cross-lyaf/enrichment_models/corrfunc_models/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_20.fits'
logM_guess = 9.5
R_guess = 1.3
logZ_guess = -3.5

# logM_guess = 9.5
# R_guess = 0.9
# logZ_guess = -3.25

linear_prior = False

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = ie.read_model_grid(modelfile)


###########################

#
# logM = params['logM'][0]
# tau_R = params['R_Mpc'][0]
# logZ = params['logZ'][0]
#
# ilogM = 2
# itau_R = 1
# ilogZ = 2
#
# covar_select = covar_array[ilogM,itau_R,ilogZ,:,:]
#
# corr = np.zeros([covar_select.shape[0],covar_select.shape[1]])
#
# for i in range(covar_select.shape[0]):
#     for j in range(covar_select.shape[1]):
#         corr[i,j] = covar_select[i,j]/np.sqrt(covar_select[i,i]*covar_select[j,j])
#
# fig = plt.figure(figsize = (10,10))
# plt.title('Correlation Matrix, logM = %.2f, R_Mpc = %.2f, logZ = %.2f' % (logM[ilogM], tau_R[itau_R], logZ[ilogZ]))
# im = plt.imshow(corr, cmap = 'jet', origin = 'lower', extent = [0,3000,0,3000])
# plt.colorbar(im)
# plt.show()


########################
#
# plt.figure(figsize=(16,8))
#
# itau_R = 1
# ilogZ = 2
#
# logMs = params['logM'][0]
# tau_Rs = params['R_Mpc'][0]
# logZs = params['logZ'][0]
#
# corr_1_1 = []
# corr_149_149 = []
# corr_100_150 = []
#
# plt.title('Correaltion in Different Parameters, R_Mpc = %.2f, logZ = %.2f' % (tau_Rs[itau_R], logZs[ilogZ]))
#
# for ilogM, logM in enumerate(logMs):
#     covar_select = covar_array[ilogM,itau_R,ilogZ,:,:]
#     corr = np.zeros([covar_select.shape[0],covar_select.shape[1]])
#
#     for i in range(covar_select.shape[0]):
#         for j in range(covar_select.shape[1]):
#             corr[i,j] = covar_select[i,j]/np.sqrt(covar_select[i,i]*covar_select[j,j])
#
#     corr_1_1.append(corr[1,1])
#     corr_149_149.append(corr[149,149])
#     corr_100_150.append(corr[100,150])
#
# plt.plot(logMs, corr_1_1, 'o-', label = '[1,1]')
# plt.plot(logMs, corr_149_149, 'o-',label = '[149,149]')
# plt.plot(logMs, corr_100_150, 'o-',label = '[100,150]')
# plt.xlabel('logM')
# plt.ylabel('Corr')
# plt.legend()
#
# plt.show()

########################


plt.figure(figsize=(16,8))

itau_R = 1
ilogM = 1

logMs = params['logM'][0]
tau_Rs = params['R_Mpc'][0]
logZs = params['logZ'][0]

corr_1_1 = []
corr_149_149 = []
corr_100_150 = []

plt.title('Correaltion in Different Parameters, R_Mpc = %.2f, logM = %.2f' % (tau_Rs[itau_R], logMs[ilogM]))

for ilogZ, logZ in enumerate(logZs):
    covar_select = covar_array[ilogM,itau_R,ilogZ,:,:]
    corr = np.zeros([covar_select.shape[0],covar_select.shape[1]])

    for i in range(covar_select.shape[0]):
        for j in range(covar_select.shape[1]):
            corr[i,j] = covar_select[i,j]/np.sqrt(covar_select[i,i]*covar_select[j,j])

    corr_1_1.append(corr[1,1])
    corr_149_149.append(corr[149,149])
    corr_100_150.append(corr[100,150])

plt.plot(logZs, corr_1_1, 'o-', label = '[1,1]')
plt.plot(logZs, corr_149_149, 'o-',label = '[149,149]')
plt.plot(logZs, corr_100_150, 'o-',label = '[100,150]')
plt.xlabel('logZ')
plt.ylabel('Corr')
plt.legend()

plt.show()
