
############### Problem ###############
problem_name  = 'compute_model_grid_CIV_lya.py'

############### Module ###############
import os
import subprocess

############### Path ###############
prob_path_local = '/Users/xinsheng/civ-cross-lyaf/igm/'

prob_path_igm = 'igm:/mnt/quasar/xinsheng/code/'

############### Files ###############
files_prob = []
files_prob.append(prob_path_local + problem_name)

############### Upload ###############
copy_command1 = ['scp'] + files_prob + [prob_path_igm]

subprocess.call(copy_command1)





#
