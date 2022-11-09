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
from scipy.optimize import curve_fit

logZ_range = np.linspace(-4.5, -2.0, 20)
tau_R_range = np.arange(0.1, 3.0+0.1, 0.3)
logM_range = np.arange(8.5, 11.0+0.1, 0.3)

########## parameters to set ##########

outpath = '/Users/xinsheng/civ-cross-lyaf/output/corr_xi_tot/'

with open(outpath + 'xi_tot.npy', 'rb') as f:

    param_tot = np.load(f)
    vel_mid_tot = np.load(f)
    xi_mean_tot = np.load(f)

logZ_num = []
xi_select = []

logM_print = 9.1
tau_R_print = 1.6

for i in range(param_tot.shape[0]):
    if np.round(param_tot[i,0],2) == logM_print and np.round(param_tot[i,1],2) == tau_R_print:
        logZ_num.append(np.round(param_tot[i,2],2))
        xi_select.append(xi_mean_tot[i])

xi_select = np.array(xi_select)
logZ_num = np.array(logZ_num)

# print(np.where(np.round(vel_mid_tot[0]) == 498)) # 97

xi_calc = np.sort(xi_select[:,97])
logZ_calc = np.sort(logZ_num)

print(xi_calc)

def fit_func_2(x, a, b, c):
    return a*x**2 + b*x + c

def fit_func_1(x, a):
    return a*x

popt_2, pcov_2 = curve_fit(fit_func_2, 10**(logZ_calc), xi_calc*1000)
popt_1, pcov_1 = curve_fit(fit_func_1, 10**(logZ_calc), xi_calc*1000)

plt.figure(figsize = (8,8))
plt.title('Relation between the xi_mean_max and logZ, for logM = %.2f and tau_R = %.2f' % (logM_print, tau_R_print))
plt.plot(10**(logZ_calc), xi_calc*1000, label = 'original curve, xi_mean_max*1000')
#plt.plot(10**(logZ_calc), fit_func_2(10**(logZ_calc), *popt_2), label = 'fit curve ax^2+bx+c, a = %.2f, b=%.2f, c=%.2f' % (popt_2[0],popt_2[1], popt_2[2]))
#plt.plot(10**(logZ_calc), fit_func_1(10**(logZ_calc), *popt_1), label = 'fit curve ax+b, a = %.2f, b=%.2f' % (popt_1[0],popt_1[1]))
plt.plot(10**(logZ_calc), fit_func_1(10**(logZ_calc), *popt_1), label = 'fit curve ax, a = %.2f' % (popt_1[0]))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Z')
plt.ylabel('xi_mean_max * 1000')
plt.legend()
plt.savefig('/Users/xinsheng/civ-cross-lyaf/output/xi_max_logZ_relation.png')

plt.close()


logM_select = np.round(param_tot[:,0],2)
R_select = np.round(param_tot[:,1],2)
logZ_select = param_tot[:,2]
xi_select_max = xi_mean_tot[:,30:100].max(axis=1)
xi_select = xi_mean_tot[:,0]
logZ_eff_array = []
xi_use = []
R_use = []
logM_use = []
fm_array = []
fv_array = []
xi_use_max = []

for i in range(len(logM_select)):
    fv, fm = CIV_lya.get_fvfm(logM_select[i], R_select[i])
    #logZ_eff = CIV_lya.calc_igm_Zeff(fm, logZ_select[i])
    logZ_eff = CIV_lya.calc_igm_Zeff(fm, logZ_select[i])
    logZ_eff_array.append(logZ_eff)
    xi_use.append(xi_select[i])
    xi_use_max.append(xi_select_max[i])
    fm_array.append(fm)
    fv_array.append(fv)




plt.figure(figsize = (8,8))
plt.title('Relation between the xi_mean_max and logZ_eff')
plt.plot(logZ_eff_array, np.array(xi_use)*1000, '.', label='max')
plt.plot(logZ_eff_array, np.array(xi_use_max)*1000, '.', label='first')
#plt.plot(10**(logZ_calc), fit_func_2(10**(logZ_calc), *popt_2), label = 'fit curve ax^2+bx+c, a = %.2f, b=%.2f, c=%.2f' % (popt_2[0],popt_2[1], popt_2[2]))
#plt.plot(10**(logZ_calc), fit_func_1(10**(logZ_calc), *popt_1), label = 'fit curve ax+b, a = %.2f, b=%.2f' % (popt_1[0],popt_1[1]))
#plt.plot(10**(logZ_calc), fit_func_1(10**(logZ_calc), *popt_1), label = 'fit curve ax, a = %.2f' % (popt_1[0]))
#plt.xscale('log')
plt.yscale('log')
plt.xlabel('logZ_eff')
plt.ylabel('xi_mean_max * 1000')
plt.legend()
plt.savefig('/Users/xinsheng/civ-cross-lyaf/output/xi_max_logZ_eff.png')

plt.close()

X = np.reshape(fv_array, (20, 90))
Y = np.reshape(logZ_select, (20, 90))
Z = np.reshape(np.array(xi_use_max), (20, 90))

# xi_use_max = []
# for i in range(len(logM_select)):
#     if logZ_select[i] == -2.0:
#         R_use.append(R_select[i])
#         logM_use.append(logM_select[i])
#         xi_use_max.append(xi_select_max[i])
#
#
# X = np.reshape(logM_use, (9, 10))
# Y = np.reshape(R_use, (9, 10))
# Z = np.reshape(np.array(xi_use_max), (9, 10))


X_sorted = []
Z_sorted = []

print(Y[0][:])

for i in range(20):
    XZ = zip(X[i][:],Z[i][:])
    XZ = sorted(XZ)
    X_sorted.append([X for X, Z in XZ])
    Z_sorted.append([Z for X, Z in XZ])

X_sorted = np.array(X_sorted)
Z_sorted = np.array(Z_sorted)

# X = np.array(fm_array)
# Y = np.array(logZ_select)
# Z = np.array(xi_use)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot_surface(X, Y, np.log10(Z))
ax.plot_surface(X_sorted, Y, np.log10(Z_sorted))
ax.set_xlabel('fv')
ax.set_ylabel('logZ_eff')
ax.set_zlabel('log(xi_max*1000)')
plt.show()
# plt.savefig('/Users/xinsheng/civ-cross-lyaf/output/fvfm.png')
# plt.close()


#
# plt.figure(figsize = (15,9))
# for i in range(xi_select.shape[0]):
#     plt.plot(vel_mid_tot[0,:], xi_select[i,:], label = 'logZ = %.2f' % logZ_num[i])
# plt.legend()
# plt.show()

# plt.figure(figsize = (15,9))
# plt.title('Relation between the xi_mean_max and logZ_eff')
# plt.plot(logZ_eff_array, np.array(xi_use)*1000, '.')
# #plt.plot(10**(logZ_calc), fit_func_2(10**(logZ_calc), *popt_2), label = 'fit curve ax^2+bx+c, a = %.2f, b=%.2f, c=%.2f' % (popt_2[0],popt_2[1], popt_2[2]))
# #plt.plot(10**(logZ_calc), fit_func_1(10**(logZ_calc), *popt_1), label = 'fit curve ax+b, a = %.2f, b=%.2f' % (popt_1[0],popt_1[1]))
# #plt.plot(10**(logZ_calc), fit_func_1(10**(logZ_calc), *popt_1), label = 'fit curve ax, a = %.2f' % (popt_1[0]))
# #plt.xscale('log')
# plt.xlabel('fv')
# plt.ylabel('fm')
# #plt.legend()
# plt.savefig('/Users/xinsheng/civ-cross-lyaf/output/fvfm.png')
#
# plt.close()
