import sys
sys.path.insert(0, "/Users/xinsheng/CIV_forest/")
sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")
sys.path.insert(0,"/Users/xinsheng/civ-cross-lyaf/code")

import inference_enrichment as ie
import pdb
from enigma.reion_forest.compute_model_grid import read_model_grid
import numpy as np
import matplotlib.pyplot as plt

modelfile = '/Users/xinsheng/civ-cross-lyaf/enrichment_models/corrfunc_models/corr_func_models_fwhm_10.000_samp_3.000_SNR_50.000_nqsos_25.fits'
logM_guess = 9.5
R_guess = 1.3
logZ_guess = -3.5

# logM_guess = 9.5
# R_guess = 0.9
# logZ_guess = -3.25

linear_prior = False

params, xi_mock_array, xi_model_array, covar_array, icovar_array, lndet_array = ie.read_model_grid(modelfile)


##########################


logM = params['logM'][0]
tau_R = params['R_Mpc'][0]
logZ = params['logZ'][0]

ilogM = 3
itau_R = 7
ilogZ = 5

print(logZ)

covar_select = covar_array[ilogM,itau_R,ilogZ,:,:]

corr = np.zeros([covar_select.shape[0]-20,covar_select.shape[1]-20])

for i in range(covar_select.shape[0]-20):
    for j in range(covar_select.shape[1]-20):
        corr[i,j] = covar_select[i,j]/np.sqrt(covar_select[i,i]*covar_select[j,j])

fig = plt.figure(figsize = (15,15))
plt.title('Correlation Matrix, logM = %.2f, R = %.2f, logZ = %.2f' % (logM[ilogM], tau_R[itau_R], logZ[ilogZ]), fontsize=30, y=1.02)
im = plt.imshow(corr, origin = 'lower', extent = [0,1000,0,1000])

cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label(label=r"$C_{ij} / \sqrt{C_{ii}C_{jj}}$",size=25)
cbar.ax.tick_params(labelsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.xlabel(r'$\Delta v$ (km/s)', fontsize=25)
plt.ylabel(r'$\Delta v$ (km/s)', fontsize=25)

plt.savefig('/Users/xinsheng/civ-cross-lyaf/present/correlation_matrix.png')



#######################
#
# plt.figure(figsize=(8,8))
#
# itau_R = 9
# ilogZ = 2
#
# logMs = params['logM'][0]
# tau_Rs = params['R_Mpc'][0]
# logZs = params['logZ'][0]
#
# corr_1_1 = []
# corr_75_25 = []
# corr_95_10 = []
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
#     corr_75_25.append(corr[75,25])
#     corr_95_10.append(corr[95,10])
#
# plt.plot(logMs, corr_1_1, 'o-', label = r'$\Delta v = 0 (km/s)$')
# plt.plot(logMs, corr_75_25, 'o-',label = r'$\Delta v = 500 (km/s)$')
# plt.plot(logMs, corr_95_10, 'o-',label = r'$\Delta v = 850 (km/s)$')
# plt.xlabel('logM')
# plt.ylabel('Corr')
# plt.legend()
#
# plt.savefig('/Users/xinsheng/civ-cross-lyaf/present/logM_relation.png')
#
# ########################
#
#
# plt.figure(figsize=(8,8))
#
# ilogM = 9
# ilogZ = 2
#
# logMs = params['logM'][0]
# tau_Rs = params['R_Mpc'][0]
# logZs = params['logZ'][0]
#
# corr_1_1 = []
# corr_75_25 = []
# corr_95_10 = []
#
# plt.title('Correaltion in Different Parameters, logM = %.2f, logZ = %.2f' % (logMs[itau_R], logZs[ilogZ]))
#
# for itau_R, tau_R in enumerate(tau_Rs):
#     covar_select = covar_array[ilogM,itau_R,ilogZ,:,:]
#     corr = np.zeros([covar_select.shape[0],covar_select.shape[1]])
#
#     for i in range(covar_select.shape[0]):
#         for j in range(covar_select.shape[1]):
#             corr[i,j] = covar_select[i,j]/np.sqrt(covar_select[i,i]*covar_select[j,j])
#
#     corr_1_1.append(corr[1,1])
#     corr_75_25.append(corr[75,25])
#     corr_95_10.append(corr[95,10])
#
# plt.plot(tau_Rs, corr_1_1, 'o-', label = r'$\Delta v = 0 (km/s)$')
# plt.plot(tau_Rs, corr_75_25, 'o-',label = r'$\Delta v = 500 (km/s)$')
# plt.plot(tau_Rs, corr_95_10, 'o-',label = r'$\Delta v = 850 (km/s)$')
# plt.xlabel('tau_R')
# plt.ylabel('Corr')
# plt.legend()
#
# plt.savefig('/Users/xinsheng/civ-cross-lyaf/present/tau_R_relation.png')
