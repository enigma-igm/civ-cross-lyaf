########## import file ##########

import sys
sys.path.insert(0, "/mnt/quasar/xinsheng/CIV_forest/")
sys.path.insert(0, "/mnt/quasar/xinsheng/enigma/enigma/reion_forest/")
sys.path.insert(0, "/mnt/quasar/xinsheng/code/")
import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import metal_corrfunc as mcf
import time
import CIV_lya_correlation as CIV_lya
import halos_skewers

########## parameters to set ##########

metal_ion = 'C IV'
fwhm = 10    # for creating the metal forest
sampling = 3 # for creating the metal forest
vmin_corr = fwhm
vmax_corr = 3000.
#dv_corr = fwhm/sampling
dv_corr = 5
snr = None # or None for noiseless data
npath = 10000
seed = 1199 # only used if npath < len(skewers)
rand = np.random.RandomState(seed)

########################################

# input and output files
# tau_R_range = np.linspace(1.0,2.0,num = 3)
# logM_range = np.linspace(9.0,9.6,num = 3)
logZ_range = np.linspace(-4.5, -2.0, 26)
tau_R_range = np.arange(0.1, 3.0, 0.1)
logM_range = np.arange(8.5, 11.0, 0.1)
# logM = '9.00'
# tau_R = '1.00'

outpath = '/mnt/quasar/xinsheng/output/corr_fit/'
out_prim = 'corr'

param_tot = []
vel_mid_tot = []
xi_mean_tot = []

for logZ in logZ_range:
    for logM in logM_range:
        for tau_R in tau_R_range:
            corr_outfile = outpath + out_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '_logZ_' + '{:.2f}'.format(logZ) + '_CIV_lya.fits'
            outcorr = Table.read(corr_outfile)
            vel_mid = outcorr['vel_mid'][0]
            xi_tot = outcorr['xi_tot']
            xi_mean = np.mean(xi_tot, axis=0)
            parameter = np.array([logM, tau_R, logZ])
            param_tot.append(parameter)
            vel_mid_tot.append(vel_mid)
            xi_mean_tot.append(xi_mean)

with open(outpath + 'xi_tot_large.npy', 'wb') as f:

    np.save(f, np.array(param_tot))
    np.save(f, np.array(vel_mid_tot))
    np.save(f,np.array(xi_mean_tot))
