import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
import sys
from enigma.tpe.skewer_tools import random_skewers
from enigma.tpe.utils import calc_eosfit2, make_tau_skewers

# Generate Random skewers for calibrating UVB
Nran = 10000 #000  # number of skewers
seed = 789
DMAX = 3000.0

zstr = 'z45'
sim_path = '/mnt/quasar/sims/L100n4096S2/'
out_path = '/mnt/quasar/xinsheng/tau/'

hdf5file = sim_path + 'z45.h5'
ranovtfile = out_path + 'rand_skewers_' + zstr + '_ovt.fits'
rantaufile = out_path + 'rand_skewers_' + zstr + '_ovt_tau.fits'

# generate ovt skewers
ret = random_skewers(Nran, hdf5file, ranovtfile, seed=seed)  # ,Ng,rand)

# Read in skewers
params = Table.read(ranovtfile, hdu=1)
skewers = Table.read(ranovtfile, hdu=2)

Nskew = len(skewers)
Ng = (skewers['ODEN'].shape)[1]

# Fit equation of state (EOS)
oden = skewers['ODEN'].reshape(Nskew * Ng)
T = skewers['T'].reshape(Nskew * Ng)
(logT0, gamma) = calc_eosfit2(oden, T, -0.5, 0.0)
params['EOS-logT0'] = logT0
params['EOS-GAMMA'] = gamma
params['seed'] = seed

GAMMA_NOW = 0.1

# generate ovt_tau skewers
#retval = make_tau_skewers(params, skewers, rantaufile, DMAX, GAMMA_UVB=GAMMA_NOW, RESCALE=False, IMPOSE_EOS=False)
retval = make_tau_skewers(params, skewers, rantaufile, DMAX, RESCALE=True, IMPOSE_EOS=False, rescale_redshift=4.5)
