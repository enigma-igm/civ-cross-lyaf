############### Problem ###############
tau_R = 1
logM = 9
############### Module ###############
import os
import subprocess
import numpy as np
############### Path ###############
datapath = "igm:/mnt/quasar/sstie/CIV_forest/Nyx_outputs/z45/enrichment_models/tau/"
savepath = "/Users/xinsheng/civ-cross-lyaf/Nyx_output/tau/"
data_pre_prim = "rand_skewers_z45_ovt_xciv_tau_R_"
data_mid_prim = "_logM_"
data_post_prim = ".fits"

tau_R_range = np.linspace(1.0,2.0,num = 3)
logM_range = np.linspace(9.0,9.6,num = 3)

for n in tau_R_range:
    for m in logM_range:
        datfile_prim = datapath + data_pre_prim + '{:.2f}'.format(n) + data_mid_prim + '{:.2f}'.format(m) + data_post_prim
        savfile_prim = savepath + data_pre_prim + '{:.2f}'.format(n) + data_mid_prim + '{:.2f}'.format(m) + data_post_prim
        if ( os.path.isfile(savfile_prim) == False ):
            copy_command_prim = ["rsync","-avzh", datfile_prim, savfile_prim]
            subprocess.call(copy_command_prim)
