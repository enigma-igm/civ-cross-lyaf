########## import file ##########

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
compute_corr = False
plottype = 'logM' # all, logZ, logM, tau_R
Zeff_label = True
# tau_R_range = np.linspace(1.0,2.0,num = 3)
# logM_range = np.linspace(9.0,9.6,num = 3)
logZ = [-3.25] # -3.5
tau_R_range = 0.9
logM_range = 9.5

# logM = '9.00'
# tau_R = '1.00'
#
CIVpath = '/Users/xinsheng/civ-cross-lyaf/Nyx_output/tau/'
lyapath = '/Users/xinsheng/civ-cross-lyaf/Nyx_output/'
lyafile = 'rand_skewers_z45_ovt_tau.fits'
data_prim = 'rand_skewers_z45_ovt_xciv'

outpath = '/Users/xinsheng/civ-cross-lyaf/output/corr_fit/'
out_prim = 'corr'
#
# tau_metal_file_CIV = datapath + data_prim + '_tau_R_' + tau_R + '_logM_' + logM + '.fits' # 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits'
# tau_metal_file_lya = datapath + 'rand_skewers_z45_ovt_tau_new.fits'
#
# corr_outfile = output + '_tau_R_' + tau_R + '_logM_' + logM + out_prim + '.fits'
# corr_outfile_blue = '/Users/xinsheng/civ-cross-lyaf/output/corr_CIV_lya_uniform_blue.fits' # saving output correlation function
# corr_outfile_red = '/Users/xinsheng/civ-cross-lyaf/output/corr_CIV_lya_uniform_red.fits'

if compute_corr and plottype != 'logZ':
    for logM in logM_range:
        for tau_R in tau_R_range:

            tau_metal_file_CIV = CIVpath + data_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '.fits' # 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits'
            tau_metal_file_lya = lyapath + lyafile
            corr_outfile = outpath + out_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '_CIV_lya.fits'

            params_CIV = Table.read(tau_metal_file_CIV, hdu=1)
            skewers_CIV = Table.read(tau_metal_file_CIV, hdu=2)

            params_lya = Table.read(tau_metal_file_lya, hdu=1)
            skewers_lya = Table.read(tau_metal_file_lya, hdu=2)

            start = time.time()
            vel_mid, xi_mean_tot, xi_tot, npix_tot = CIV_lya.compute_xi_all_CIV_lya(params_CIV, skewers_CIV, params_lya, skewers_lya, logZ, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr=snr, sampling=sampling)
            mcf.write_corrfunc(vel_mid, xi_tot, npix_tot, corr_outfile)
            end = time.time()

            print("Done computing 2PCF in %0.2f min" % ((end-start)/60.))

elif compute_corr and plottype == 'logZ':
    for logZ_value in logZ:
        tau_R = tau_R_range
        logM = logM_range
        tau_metal_file_CIV = CIVpath + data_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '.fits' # 'nyx_sim_data/rand_skewers_z45_ovt_tau_xciv_flux.fits'
        tau_metal_file_lya = lyapath + lyafile
        corr_outfile = outpath + out_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '_logZ_' + '{:.2f}'.format(logZ_value) + '_CIV_lya.fits'

        params_CIV = Table.read(tau_metal_file_CIV, hdu=1)
        skewers_CIV = Table.read(tau_metal_file_CIV, hdu=2)

        params_lya = Table.read(tau_metal_file_lya, hdu=1)
        skewers_lya = Table.read(tau_metal_file_lya, hdu=2)

        start = time.time()
        vel_mid, xi_mean_tot, xi_tot, npix_tot = CIV_lya.compute_xi_all_CIV_lya(params_CIV, skewers_CIV, params_lya, skewers_lya, logZ_value, fwhm, metal_ion, vmin_corr, vmax_corr, dv_corr, snr=snr, sampling=sampling)
        mcf.write_corrfunc(vel_mid, xi_tot, npix_tot, corr_outfile)
        end = time.time()

        print("Done computing 2PCF in %0.2f min" % ((end-start)/60.))

else:
    if plottype == 'all':
        plt.figure(figsize=(12,8))
        for logM in logM_range:
            for tau_R in tau_R_range:
                corr_outfile = outpath + out_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '_CIV_lya.fits'
                label = 'tau_R = ' + '{:.2f}'.format(tau_R) + ', logM = ' + '{:.2f}'.format(logM)
                outcorr = Table.read(corr_outfile)
                vel_mid = outcorr['vel_mid'][0]
                xi_tot = outcorr['xi_tot']
                xi_mean_tot = np.mean(xi_tot, axis=0)
                plt.plot(vel_mid, xi_mean_tot, linewidth=2.0, label = label)
        plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)

        vel_doublet = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.title('%d skewers, fwhm=%d km/s, sampling=%d, logZ = %0.1f' % (len(xi_tot), fwhm, sampling, logZ) + \
                  '\n' + 'vmin = %0.1f, vmax=%0.1f, dv=%0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)
        plt.legend(frameon=False)
        plt.xlim([-50, 2000])
        plt.savefig(outpath + 'coor_CIV_lya_all.pdf')

    elif plottype == 'logM':
        for logM in logM_range:
            plt.figure(figsize=(12,8))
            for tau_R in tau_R_range:
                corr_outfile = outpath + out_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '_CIV_lya.fits'
                if Zeff_label == True:
                    fv, fm = CIV_lya.get_fvfm(logM, tau_R)
                    Zeff = CIV_lya.calc_igm_Zeff(fm, logZ)
                    label = 'tau_R = ' + '{:.2f}'.format(tau_R) + ', logM = ' + '{:.2f}'.format(logM) + ', log_Zeff = ' + '{:.2f}'.format(Zeff)
                else:
                    label = 'tau_R = ' + '{:.2f}'.format(tau_R) + ', logM = ' + '{:.2f}'.format(logM)
                outcorr = Table.read(corr_outfile)
                vel_mid = outcorr['vel_mid'][0]
                xi_tot = outcorr['xi_tot']
                xi_mean_tot = np.mean(xi_tot, axis=0)
                plt.plot(vel_mid, xi_mean_tot, linewidth=2.0, label = label)
            plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
            plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)

            vel_doublet = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
            plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
            plt.title('%d skewers, fwhm=%d km/s, sampling=%d, logZ = %0.1f' % (len(xi_tot), fwhm, sampling, logZ) + \
                    '\n' + 'vmin = %0.1f, vmax=%0.1f, dv=%0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)
            plt.legend(frameon=False)
            plt.xlim([-50, 2000])
            plt.savefig(outpath + 'coor_CIV_lya' + '_logM_' + '{:.2f}'.format(logM) + '.pdf')

    elif plottype == 'tau_R':
        for tau_R in tau_R_range:
            plt.figure(figsize=(12,8))
            for logM in logM_range:
                corr_outfile = outpath + out_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '_CIV_lya.fits'
                if Zeff_label == True:
                    fv, fm = CIV_lya.get_fvfm(logM, tau_R)
                    Zeff = CIV_lya.calc_igm_Zeff(fm, logZ)
                    label = 'tau_R = ' + '{:.2f}'.format(tau_R) + ', logM = ' + '{:.2f}'.format(logM) + ', logZ_eff = ' + '{:.2f}'.format(Zeff)
                else:
                    label = 'tau_R = ' + '{:.2f}'.format(tau_R) + ', logM = ' + '{:.2f}'.format(logM)
                outcorr = Table.read(corr_outfile)
                vel_mid = outcorr['vel_mid'][0]
                xi_tot = outcorr['xi_tot']
                xi_mean_tot = np.mean(xi_tot, axis=0)
                plt.plot(vel_mid, xi_mean_tot, linewidth=2.0, label = label)
            plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
            plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)

            vel_doublet = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
            plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
            plt.title('%d skewers, fwhm=%d km/s, sampling=%d, logZ = %0.1f' % (len(xi_tot), fwhm, sampling, logZ) + \
                    '\n' + 'vmin = %0.1f, vmax=%0.1f, dv=%0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)
            plt.legend(frameon=False)
            plt.xlim([-50, 2000])
            plt.savefig(outpath + 'coor_CIV_lya' + '_tau_R_' + '{:.2f}'.format(tau_R) + '.pdf')

    elif plottype == 'logZ':
        plt.figure(figsize=(12,8))
        tau_R = tau_R_range
        logM = logM_range
        for logZ_value in logZ:
            corr_outfile = outpath + out_prim + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '_logZ_' + '{:.2f}'.format(logZ_value) + '_CIV_lya.fits'
            if Zeff_label == True:
                fv, fm = CIV_lya.get_fvfm(logM, tau_R)
                #logZ = -3.25
                Zeff = CIV_lya.calc_igm_Zeff(fm, logZ)
                label = label = 'logZ = ' + '{:.2f}'.format(logZ_value) + ', logZ_eff = ' + '{:.2f}'.format(Zeff)
            else:
                label = 'logZ = ' + '{:.2f}'.format(logZ_value)
            outcorr = Table.read(corr_outfile)
            vel_mid = outcorr['vel_mid'][0]
            xi_tot = outcorr['xi_tot']
            xi_mean_tot = np.mean(xi_tot, axis=0)
            plt.plot(vel_mid, xi_mean_tot, linewidth=2.0, label = label)
        plt.xlabel(r'$\Delta v$ (km/s)', fontsize=15)
        plt.ylabel(r'$\xi(\Delta v)$', fontsize=15)

        vel_doublet = reion_utils.vel_metal_doublet(metal_ion, returnVerbose=False)
        plt.axvline(vel_doublet.value, color='red', linestyle=':', linewidth=1.2, label='Doublet separation (%0.1f km/s)' % vel_doublet.value)
        plt.title('%d skewers, fwhm=%d km/s, sampling=%d, ' % (len(xi_tot), fwhm, sampling) + \
                '\n' + 'vmin = %0.1f, vmax=%0.1f, dv=%0.1f' % (vmin_corr, vmax_corr, dv_corr), fontsize=15)
        plt.legend(frameon=False)
        plt.xlim([-50, 2000])
        plt.savefig(outpath + 'coor_CIV_lya' + '_tau_R_' + '{:.2f}'.format(tau_R) + '_logM_' + '{:.2f}'.format(logM) + '.pdf')
