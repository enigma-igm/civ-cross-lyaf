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
import subprocess
from multiprocessing import Pool

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

logZ_range = np.round(np.linspace(-4.5, -2.0, 26), 2)
tau_R_range = np.round(np.arange(0.1, 3.0, 0.1), 2)
logM_range = np.round(np.arange(8.5, 11.0, 0.1), 2)

CIVpath = '/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/tau/'
lyapath = '/mnt/quasar/xinsheng/output/lya_forest/'
lyafile = 'rand_skewers_z45_ovt_tau.fits'
data_prim = 'rand_skewers_z45_ovt_xciv'

outpath = '/mnt/quasar/xinsheng/output/corr_fit/'
out_prim = 'corr'

def correlation_file_generator(logZ, logM, tau_R):
    tau_metal_file_CIV = CIVpath + data_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '.fits' # 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits'
    tau_metal_file_lya = lyapath + lyafile
    corr_outfile = outpath + out_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '_logZ_' + '{:.2f}'.format(logZ) + '_CIV_lya.fits'

    params_CIV = Table.read(tau_metal_file_CIV, hdu=1)
    skewers_CIV = Table.read(tau_metal_file_CIV, hdu=2)

    params_lya = Table.read(tau_metal_file_lya, hdu=1)
    skewers_lya = Table.read(tau_metal_file_lya, hdu=2)

    start = time.time()

    vel_mid, xi_mean_tot, xi_tot, npix_tot = CIV_lya.compute_xi_all_CIV_lya(params_CIV, skewers_CIV, params_lya, skewers_lya, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr=snr, sampling=sampling)
    mcf.write_corrfunc(vel_mid, xi_tot, npix_tot, corr_outfile)

    end = time.time()

    print("Done computing 2PCF in %0.2f min" % ((end-start)/60.))

def multiproc(start_num):
    for logM in logM_range:
        for tau_R in tau_R_range:
            logZ = logZ_range[start_num]
            correlation_file_generator(logZ, logM, tau_R)
            #print('processor %d finished %d out of %d' % (start_num, counter_solve-start_num, logM_range.size * tau_R_range.size))


p = Pool(10)
p.map(multiproc, range(0,10))
p.close()
p.join()
