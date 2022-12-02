
############### Problem ###############
# problem_name1  = 'likelihood_auto.job'
# problem_name2  = 'likelihood.job'
problem_name1  = 'covar_generator.job'
problem_name2  = 'covar_generator.py'
problem_name3  = 'likelihood.job'
problem_name4  = 'plot_mockdata.py'
problem_name5 = 'CIV_lya_correlation.py'

############### Module ###############
import os
import subprocess

############### Path ###############
prob_path_local = '/Users/xinsheng/civ-cross-lyaf/pod/code/'

prob_path_pod = 'pod:/home/xinsheng/enigma/code/'

############### Files ###############
files_prob1 = []
files_prob2 = []
files_prob3 = []
files_prob4 = []
files_prob5 = []

files_prob1.append(prob_path_local + problem_name1)
files_prob2.append(prob_path_local + problem_name2)
files_prob3.append(prob_path_local + problem_name3)
files_prob4.append(prob_path_local + problem_name4)
files_prob5.append(prob_path_local + problem_name5)

############### Upload ###############
copy_command1 = ['scp'] + files_prob1 + [prob_path_pod]
copy_command2 = ['scp'] + files_prob2 + [prob_path_pod]
copy_command3 = ['scp'] + files_prob3 + [prob_path_pod]
copy_command4 = ['scp'] + files_prob4 + [prob_path_pod]
copy_command5 = ['scp'] + files_prob5 + [prob_path_pod]

subprocess.call(copy_command1)
subprocess.call(copy_command2)
subprocess.call(copy_command3)
subprocess.call(copy_command4)
subprocess.call(copy_command5)



#
