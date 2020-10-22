"""
crontab -e: to plan this command to be executed regurarly
help on crontab: https://www.cyberciti.biz/faq/how-do-i-add-jobs-to-cron-under-linux-or-unix-oses/
"""
import subprocess
import sys
import os
import pandas as pd

file_with_jobs = '/home/bgonchar/correlated_noise_pta_2020/data/dr2_timing_20200607/corr_noise_jobs.xlsx'

list_slurm_job_names = ['squeue','--user','bgonchar','--Format=name:50','--noheader']

command_output = subprocess.check_output(list_slurm_job_names)
if int(sys.version[0]) >= 3:
  command_output = command_output.decode(sys.stdout.encoding)

active_jobs = command_output.replace(' ', '').split('\n')

noted_jobs = pd.read_excel(file_with_jobs)

idx_completed = list()

for index, row in noted_jobs.iterrows():

  jj = row['job_name']
  ss = row['slurm_file_name']
  rr = row['result_file_name']

  if jj not in active_jobs:
    if not os.path.isfile(rr):
      print('Re-submitting job ',jj)
      subprocess.call(['sbatch',ss])
    else:
      idx_completed.append(index)
      print('Job ',jj,' completed')
  else:
    print('Job ',jj,' is still running or in the queue')

noted_jobs = noted_jobs.drop(index = idx_completed)
noted_jobs.to_excel(file_with_jobs, engine='openpyxl', index=False)
