import sys
sys.path.insert(0, "/Users/xinsheng/CIV_forest/")
sys.path.insert(0, "/Users/xinsheng/enigma/enigma/reion_forest/")

import numpy as np
import matplotlib.pyplot as plt
import enigma.reion_forest.utils as reion_utils
from astropy.table import Table
import metal_corrfunc as mcf
import time
from astropy.io import fits

filename = '/Users/xinsheng/civ-cross-lyaf/pod/output/inference_test_covar/save_mid_logM_9.90_R_1.00_logZ_-3.60_k_3.npy'
savefig = '/Users/xinsheng/civ-cross-lyaf/pod/output/inference_test_covar/inference_test_new.pdf'

with open(filename, 'rb') as f:
    logM = np.load(f)
    R = np.load(f)
    logZ = np.load(f)

logM = np.reshape(logM, logM.size)
R = np.reshape(R, R.size)
logZ = np.reshape(logZ, logZ.size)

plt.figure(figsize = (15,5))
plt.suptitle('Inference test for logM = 9.9, R = 1.0 and logZ = -3.6')
plt.subplot(1,3,1)
plt.hist(logM, bins=26)
plt.xlabel('logM')
plt.ylabel('Count')
plt.subplot(1,3,2)
plt.hist(R, bins=30)
plt.xlabel('R')
plt.ylabel('Count')
plt.subplot(1,3,3)
plt.hist(logZ, bins=26)
plt.xlabel('logZ')
plt.ylabel('Count')
plt.legend()
plt.savefig(savefig)
plt.close()
