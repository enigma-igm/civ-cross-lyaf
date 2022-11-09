
############### Problem ###############
problem_name1  = 'corner.pdf'
problem_name2  = 'fit.pdf'
problem_name3  = 'probability_cov.png'
problem_name4  = 'fit_fv_logZ.pdf'
problem_name5  = 'corner_fv_logZ.pdf'

############### Module ###############
import os
import subprocess

############### Path ###############
prob_path_local = 'pod:/home/xinsheng/enigma/output/mcmc_compare/'

prob_path_pod = '/Users/xinsheng/civ-cross-lyaf/pod/output/mcmc_compare/'

# prob_path_local = 'pod:/home/xinsheng/enigma/output/auto/mcmc/'
#
# prob_path_pod = '/Users/xinsheng/civ-cross-lyaf/pod/output/auto/mcmc/'

############### Files ###############
files_prob1 = []
files_prob2 = []
files_prob3 = []
files_prob4 = []
files_prob5 = []
#files_prob4 = []

files_prob1.append(prob_path_local + problem_name1)
files_prob2.append(prob_path_local + problem_name2)
files_prob3.append(prob_path_local + problem_name3)
files_prob4.append(prob_path_local + problem_name4)
files_prob5.append(prob_path_local + problem_name5)
#files_prob4.append(prob_path_local + problem_name4)

############### Upload ###############
copy_command1 = ['scp'] + files_prob1 + [prob_path_pod]
copy_command2 = ['scp'] + files_prob2 + [prob_path_pod]
copy_command3 = ['scp'] + files_prob3 + [prob_path_pod]
copy_command4 = ['scp'] + files_prob4 + [prob_path_pod]
copy_command5 = ['scp'] + files_prob5 + [prob_path_pod]
#copy_command4 = ['scp'] + files_prob4 + [prob_path_pod]

subprocess.call(copy_command1)
subprocess.call(copy_command2)
subprocess.call(copy_command3)
subprocess.call(copy_command4)
subprocess.call(copy_command5)
#subprocess.call(copy_command4)




#
