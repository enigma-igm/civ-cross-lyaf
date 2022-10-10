"""
Plot oden, T, x_metal, and flux skewers for a randomly selected skewer.
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from astropy.table import Table
import sys

sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")
import enigma.reion_forest.utils as reion_utils


sys.path.insert(0, "/Users/xinsheng/CIV_forest/")

#from enigma.reion_forest import utils as reion_utils
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import halos_skewers

sys.path.insert(0,"/Users/xinsheng/civ-cross-lyaf/code")
import CIV_lya_correlation as CIV_lya

# TODO: Plot enrichment topology (i.e. mask skewer in the xciv panel) and Skewer of N_CIV.

### Figure settings
font = {'family' : 'serif', 'weight' : 'normal'}#, 'size': 11}
plt.rc('font', **font)
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 4

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(16, 12), sharex=True)
fig.subplots_adjust(left=0.1, bottom=0.07, right=0.98, top=0.93, wspace=0, hspace=0.)

xytick_size = 16
xylabel_fontsize = 20
legend_fontsize = 14
linewidth = 2.5
alpha_uniform = 0.6
alpha_data = 0.5

### Files
#data_path = '/Users/xinsheng/XSWork/CIV/Nyx_output/'
data_path = '/Users/xinsheng/'

skewerfile_CIV = '/Users/xinsheng/civ-cross-lyaf/Nyx_output/tau/rand_skewers_z45_ovt_xciv_tau_R_1.30_logM_9.70.fits'

skewerfile = '/Users/xinsheng/civ-cross-lyaf/Nyx_output/rand_skewers_z45_ovt_tau.fits'


metal_par = Table.read(skewerfile, hdu=1)
metal_ske = Table.read(skewerfile, hdu=2)

metal_par_CIV = Table.read(skewerfile_CIV, hdu=1)
metal_ske_CIV = Table.read(skewerfile_CIV, hdu=2)

logM = float(skewerfile_CIV.split('logM_')[-1].split('.fits')[0])
R_Mpc = float(skewerfile_CIV.split('R_')[-1].split('_logM')[0])
logZ = -3.5
savefig = '/Users/xinsheng/civ-cross-lyaf/figure/CIV_skewers.png'
#savefig = '/Users/xinsheng/XSWork/CIV/figure/skewers_R_%0.2f_logM_%0.2f.pdf' % (logM, R_Mpc)

metal_ion = 'C IV'
fwhm = 10 # km/s
snr = 50
sampling = 3

# cosmology
z = metal_par['z'][0]
cosmo = FlatLambdaCDM(H0=100.0 * metal_par['lit_h'][0], Om0=metal_par['Om0'][0], Ob0=metal_par['Ob0'][0])
Hz = (cosmo.H(z))
a = 1.0 / (1.0 + z)

#i = np.random.randint(0, len(metal_ske))
i = 2500 # other good los: 4197, 7504, 1061
print('random index', i)

# creating the metal forest for random skewer 'i'
v_lores, (ftot_lores, figm_lores, fcgm_lores), \
v_hires, (ftot_hires, figm_hires, fcgm_hires), \
(oden, T, x_metal), cgm_tup, tau_igm = CIV_lya.create_lya_forest(metal_par, metal_ske[[i]], fwhm, sampling=sampling)

v_lores_CIV, (ftot_lores_CIV, figm_lores_CIV, fcgm_lores_CIV), \
v_hires_CIV, (ftot_hires_CIV, figm_hires_CIV, fcgm_hires_CIV), \
(oden_CIV, v_los_CIV, T_CIV, x_metal_CIV), cgm_tup_CIV, tau_CIV = CIV_lya.create_metal_forest_tau(metal_par_CIV, metal_ske_CIV[[i]], logZ, fwhm, metal_ion, sampling=sampling)

print(v_lores.min(), v_hires.min(), v_lores.max(), v_hires.max())

# # ~0.00014 sec to generate one skewer

#### uniformly enriched ####

# ori_skewerfile = data_path + 'tau/rand_skewers_z45_ovt_xciv_tau_R_1.40_logM_10.50.fits' # uniformly enriched
# #ori_skewerfile = 'nyx_sim_data/tmp_igm_cluster/rand_skewers_z45_ovt_xciv_R_1.35_logM_11.00_tau.fits'
# ori_metal_par = Table.read(ori_skewerfile, hdu=1)
# ori_metal_ske = Table.read(ori_skewerfile, hdu=2)
#
# ori_v_lores, (ori_ftot_lores, ori_figm_lores, ori_fcgm_lores), \
# ori_v_hires, (ori_ftot_hires, ori_figm_hires, ori_fcgm_hires), \
# (ori_oden, ori_v_los, ori_T, ori_x_metal), ori_cgm_tup = reion_utils.create_metal_forest(ori_metal_par, ori_metal_ske[[i]], logZ, fwhm, metal_ion)
# ###########################
tau = metal_ske['TAU'][i]
vmin, vmax = v_hires.min(), v_hires.max()

# Add noise
noise = np.random.normal(0.0, 1.0/snr, ftot_lores[0].flatten().shape)
ftot_lores_noise = ftot_lores[0] + noise

noise_CIV = np.random.normal(0.0, 1.0/snr, ftot_lores_CIV[0].flatten().shape)
ftot_lores_noise_CIV = ftot_lores_CIV[0] + noise_CIV

#### oden plot ####
ax1.plot(v_hires_CIV, oden_CIV[0], c='r')
#ax1.set_ylabel('Overdensity', fontsize=xylabel_fontsize)
ax1.set_ylabel(r'$\Delta$ [$\rho/\bar{\rho}$]', fontsize=xylabel_fontsize)
ax1.tick_params(top=True, which='both', labelsize=xytick_size)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
oden_min, oden_max = -2, np.round(2 + oden[0].max())
ax1.set_xlim([vmin, vmax])
ax1.set_ylim([oden_min, oden_max])
### tau plot ###

ax2.plot(v_hires, np.log10(tau_igm[0]), c='k')
#ax1.set_ylabel('Overdensity', fontsize=xylabel_fontsize)
ax2.set_ylabel(r'$log(\tau_{H})$', fontsize=xylabel_fontsize)
ax2.tick_params(top=True, which='both', labelsize=xytick_size)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.set_xlim([vmin, vmax])

ax3.plot(v_hires, tau_CIV[0], c='r')
#ax1.set_ylabel('Overdensity', fontsize=xylabel_fontsize)
ax3.set_ylabel(r'$\tau_{CIV}$', fontsize=xylabel_fontsize)
ax3.tick_params(top=True, which='both', labelsize=xytick_size)
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.yaxis.set_minor_locator(AutoMinorLocator())
ax3.set_xlim([vmin, vmax])

print(len(v_hires))
#### flux plot ####
#ax3.plot(ori_v_hires, ori_ftot_hires[0], alpha=0.7, label='hires (uniform Z)')#, drawstyle='steps-mid', alpha=0.6, zorder=10, color='red')
ax4.plot(v_hires, ftot_hires[0], 'k', drawstyle='steps-mid')#, alpha=0.6, zorder=10, color='red')
ax4.plot(v_lores, ftot_lores_noise, label='FWHM=%0.1f km/s; SNR=%0.1f' % (fwhm, snr), c='r', alpha=alpha_data, zorder=1, drawstyle='steps-mid')
ax4.set_xlabel('v [km/s]', fontsize=xylabel_fontsize)
ax4.set_ylabel(r'F$_{\mathrm{H}}$', fontsize=xylabel_fontsize)
ax4.legend(fontsize=legend_fontsize, ncol=2, loc=1)
ax4.set_xlim([vmin, vmax])
ax4.set_ylim([-0.2, 1.4])
ax4.tick_params(top=True, which='both', labelsize=xytick_size)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())

ax5.annotate('log(M)={:5.2f} '.format(logM) + r'M$_{\odot}$, ' + 'R={:5.2f} cMpc, '.format(R_Mpc) + '[C/H]=${:5.2f}$'.format(logZ), \
             xy=(500, 1.185), xytext=(500, 1.085), textcoords='data', xycoords='data', annotation_clip=False, fontsize=legend_fontsize)
ax5.plot(v_hires_CIV, ftot_hires_CIV[0], 'k', drawstyle='steps-mid')#, alpha=0.6, zorder=10, color='red')
ax5.plot(v_lores_CIV, ftot_lores_noise_CIV, label='FWHM=%0.1f km/s; SNR=%0.1f' % (fwhm, snr), c='r', alpha=alpha_data, zorder=1, drawstyle='steps-mid')
ax5.set_xlabel('v [km/s]', fontsize=xylabel_fontsize)
ax5.set_ylabel(r'F$_{\mathrm{CIV}}$', fontsize=xylabel_fontsize)
ax5.legend(fontsize=legend_fontsize, ncol=2, loc=1)
ax5.set_xlim([vmin, vmax])
ax5.set_ylim([0.9, 1.12])
ax5.tick_params(top=True, which='both', labelsize=xytick_size)
ax5.xaxis.set_minor_locator(AutoMinorLocator())
ax5.yaxis.set_minor_locator(AutoMinorLocator())

# plot upper axis
rmin = (vmin*u.km/u.s/a/Hz).to('Mpc').value
rmax = (vmax*u.km/u.s/a/Hz).to('Mpc').value
atwin = ax1.twiny()
atwin.set_xlabel('R [cMpc]', fontsize=xylabel_fontsize, labelpad=8)
atwin.axis([rmin, rmax, oden_min, oden_max])
atwin.tick_params(top=True, axis="x", labelsize=xytick_size)
atwin.xaxis.set_minor_locator(AutoMinorLocator())

plt.savefig(savefig)
plt.close()
