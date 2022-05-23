
############### Problem ###############
problem_id  = 'mag_acc_v12'

############### Module ###############
import os
import subprocess

############### Path ###############
home_path_local = '/Users/xinsheng/XSwork/pod/'
work_path_local = home_path_local
prob_path_local = work_path_local + problem_id + '/'

home_path_pod = 'pod:/home/xinsheng/'
work_path_pod = home_path_pod
prob_path_pod = work_path_pod + problem_id + '/'

############### Files ###############
files_prob = []
#files_prob.append( prob_path_local + 'athinput.' + problem_id )
#files_prob.append( prob_path_local + problem_id + '.cpp' )
#files_prob.append( prob_path_local + problem_id + '.py' )
files_prob.append( prob_path_local + problem_id + '.job' )
#files_ini = []
#files_ini.append( prob_path_local + 'ini/ini_condtion.txt')
#files_ini.append( prob_path_local + 'ini/mag_opacity.txt')
#files_ini.append( prob_path_local + 'ini/sim_parameter.pkl')

############### Upload ###############
copy_command1 = ['scp'] + files_prob + [prob_path_pod]
#copy_command2 = ['scp'] + files_ini + [prob_path_pod+'ini/']
subprocess.call(copy_command1)
#subprocess.call(copy_command2)




#
