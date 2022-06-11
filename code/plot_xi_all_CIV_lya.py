import sys
sys.path.insert(0, "/Users/xinsheng/CIV_forest/")
sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")
sys.path.insert(0,"/Users/xinsheng/civ-cross-lyaf/code")

import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import metal_corrfunc as mcf
import time
import CIV_lya_correlation as CIV_lya

# parameters to set
metal_ion = 'C IV'
logZ = -3.5
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

# input and output files
tau_metal_file_CIV = '/Users/xinsheng/civ-cross-lyaf/Nyx_output/tau/rand_skewers_z45_ovt_xciv_tau_R_1.00_logM_9.30.fits' # 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits'
tau_metal_file_lya = '/Users/xinsheng/civ-cross-lyaf/Nyx_output/rand_skewers_z45_ovt_tau_new.fits'
corr_outfile = '/Users/xinsheng/civ-cross-lyaf/output/corr_tau_R_1.00_logM_9.30_CIV_auto.fits' # saving output correlation function
compute_corr = False

if compute_corr:

    params_CIV = Table.read(tau_metal_file_CIV, hdu=1)
    skewers_CIV = Table.read(tau_metal_file_CIV, hdu=2)

    params_lya = Table.read(tau_metal_file_lya, hdu=1)
    skewers_lya = Table.read(tau_metal_file_lya, hdu=2)

    if npath < len(skewers_CIV):
        print('randomly selecting %d skewers...' % npath)
        indx = rand.choice(len(skewers_CIV), replace=False, size=npath)
        skewers_lya = skewers_lya[indx]
        skewers_CIV = skewers_CIV[indx]

    start = time.time()
    vel_mid, xi_mean_tot, xi_tot, npix_tot = CIV_lya.compute_xi_all_CIV_lya(params_CIV, skewers_CIV, params_lya, skewers_lya, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr=snr, sampling=sampling)
    mcf.write_corrfunc(vel_mid, xi_tot, npix_tot, corr_outfile)
    end = time.time()

    print("Done computing 2PCF in %0.2f min" % ((end-start)/60.))

else:
    outcorr = Table.read(corr_outfile)
    vel_mid = outcorr['vel_mid'][0]
    xi_tot = outcorr['xi_tot']
    xi_mean_tot = np.mean(xi_tot, axis=0)

    factor = 1.0
    plt.figure(figsize=(12,8))
    plt.plot(vel_mid, factor*xi_mean_tot, linewidth=2.0, linestyle='-')#, label='SNR=%d' % snr)
    #plt.plot(vel_mid_noiseless, factor*xi_mean_tot_noiseless, linewidth=2.0, linestyle='-', color='k', alpha=0.6, label='Noiseless')
    plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
    plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)

    ymin, ymax = (factor*xi_mean_tot).min(), 1.07*((factor*xi_mean_tot).max())
    vel_doublet = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
    plt.title('%d skewers, fwhm=%d km/s, sampling=%d, logZ = %0.1f' % (len(xi_tot), fwhm, sampling, logZ) + \
              '\n' + 'vmin = %0.1f, vmax=%0.1f, dv=%0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)
    plt.legend(frameon=False)
    plt.xlim([-50, 2000])
    plt.ylim([ymin, ymax])
    plt.savefig('/Users/xinsheng/civ-cross-lyaf/output/coor_CIV_tau_R_1.00_logM_9.30_auto.pdf')
