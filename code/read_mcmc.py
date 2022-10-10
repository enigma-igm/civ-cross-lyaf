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

filename = '/Users/xinsheng/civ-cross-lyaf/enrichment_models/corrfunc_models/mcmc_chain_Fig13.fits'

hdu = fits.open(filename)
