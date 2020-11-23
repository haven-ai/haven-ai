import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

from haven.haven_jobs import slurm_manager as sm

  
from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import job_configs
import time
import numpy as np
from torch.utils.data import RandomSampler, DataLoader


# Job submission
# ==============
def submit_job(api, account_id, command, job_config, workdir, savedir_logs=None):
    job_spec = get_job_spec(job_config, command, savedir_logs, workdir=workdir)
    
    # read slurm setting
    lines = "#! /bin/bash \n"
    lines += "#SBATCH --account=%s \n" % job_configs.ACCOUNT_ID
    # todo: check how the job_spec is defined
    for key in list(job_spec.JOB_CONFIG.keys()):
        lines += "#SBATCH --%s=%s \n" % (key, job_spec.JOB_CONFIG[key])
    lines += command

    exp_id = command.split("-ei ")[1].split(" ")[0]  
    file_name = "%s.sh" % exp_id
    hu.save_txt(file_name, lines)
    # launch the exp
    submit_command = "sbatch %s.sh" % exp_id
    job_id = hu.subprocess_call(submit_command).split()[-1]

    # delete the slurm.sh
    os.remove(file_name)
    return job_id

def get_job_spec(job_config, command, savedir_logs, workdir):
    _job_config = copy.deepcopy(job_config)
    _job_config['workdir'] = workdir
    
    if savedir_logs is not None:
        path_log = os.path.join(savedir_logs, "logs.txt")
        path_err = os.path.join(savedir_logs, "err.txt")
        command_with_logs = '%s 1>%s 2>%s' % (command, path_log, path_err)
    else:
        command_with_logs = command

    _job_config['command'] = ['/bin/bash', '-c', command_with_logs]
    _job_config['resources'] = eai_toolkit_client.JobSpecResources(**_job_config['resources'])
    job_spec = eai_toolkit_client.JobSpec(**_job_config)

    # Return the Job command in Byte format
    return job_spec


# Job status
# ===========
def get_job(api, job_id):
    """Get job information."""
    try:
        return api.v1_job_get_by_id(job_id)
    except ApiException as e:
        raise ValueError("job id %s not found." % job_id)

def get_jobs(api, account_id):
    # account_id = hu.subprocess_call('eai account get').split('\n')[-2].split(' ')[0]
    return api.v1_account_job_get(account_id=account_id,
            limit=1000, 
            order='-created',
            q="alive_recently=True").items
            

    # return api.v1_me_job_get(limit=1000, 
    #         order='-created',
    #         q="alive_recently=True").items
           

# Job kill
# ===========
def kill_job(api, job_id):
    """Kill a job job until it is dead."""
    job = get_job(api, job_id)

    if not job.alive:
        print('%s is already dead' % job_id)
    else:
        # toolkit
        api.v1_job_delete_by_id(job_id)
        print('%s CANCELLING...' % job_id)
        job = get_job(api, job_id)
        while job.state == "CANCELLING":
            time.sleep(2.0)
            job = get_job(api, job_id)

        print('%s now is dead.' % job_id)

import argparse


def get_existing_slurm_job_commands(exp_list, savedir_base):
    existing_job_commands = []
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)
        file_name = os.path.join(savedir, "job_dict.json")
        if not os.path.exists(file_name):
            continue
        job_dict = hu.load_json(file_name)
        job_id = job_dict["job_id"]
        job_status = hu.subprocess_call("scontrol show job %s" % job_id).split("JobState=")[1].split(" ")[0]
        if job_status == "RUNNING" or job_status == "PENDING":
            existing_job_commands += [job_dict["command"]]
        
    return existing_job_commands


def launch_slurm_job(command, savedir_base):
    # read slurm setting
    lines = "#! /bin/bash \n"
    lines += "#SBATCH --account=%s \n" % job_configs.ACCOUNT_ID
    for key in list(job_configs.JOB_CONFIG.keys()):
        lines += "#SBATCH --%s=%s \n" % (key, job_configs.JOB_CONFIG[key])
    lines += command

    exp_id = command.split("-ei ")[1].split(" ")[0]  
    file_name = "%s.sh" % exp_id
    hu.save_txt(file_name, lines)
    # launch the exp
    submit_command = "sbatch %s" % file_name
    job_id = hu.subprocess_call(submit_command).split()[-1]

    # save the command and job id in job_dict.json
    job_dict = {
        "command": command,
        "job_id": job_id
    }
    savedir = os.path.join(savedir_base, exp_id)
    hu.save_json(os.path.join(savedir, "job_dict.json"), job_dict)

    # delete the slurm.sh
    os.remove(file_name)


if __name__ == "__main__":
  # run with job scheduler
  exp_list = [{'model':{'name':'mlp', 'n_layers':20}, 
              'dataset':'mnist', 'batch_size':1}]
  command_list = []
  for exp_dict in exp_list:
      save_exp_folder(exp_dict, args.savedir_base, args.reset)
      exp_id = hu.hash_dict(exp_dict)
      command_list += ["echo 3"]
  # get slurm existing commands
  existing_commands = get_existing_slurm_job_commands(exp_list, args.savedir_base)
  for command in command_list:
      # check if command exists
      if command in existing_commands:
          print('command exists')
          continue
      # otherwise launch command
      launch_slurm_job(command, args.savedir_base) 



  



  # jm = hjb.JobManager(exp_list=exp_list, 
  #                 savedir_base=savedir_base, 
  #                 workdir=os.path.dirname(os.path.realpath(__file__)),
  #                 job_config=job_config,
  #                 )
  # # get jobs              
  # job_list_old = jm.get_jobs()

  # # run single command
  # savedir_logs = '%s/%s' % (savedir_base, np.random.randint(1000))
  # os.makedirs(savedir_logs, exist_ok=True)
  # command = 'echo 2'
  # job_id = jm.submit_job(command,  workdir=jm.workdir, savedir_logs=savedir_logs)

  # # get jobs
  # job_list = jm.get_jobs()
  # job = jm.get_job(job_id)
  # assert job_list[0].id == job_id
  
  # # jm.kill_job(job_list[0].id)
  # # run
  # print('jobs:', len(job_list_old), len(job_list))
  # assert (len(job_list_old) + 1) ==  len(job_list)

  # # command_list = []
  # # for exp_dict in exp_list:
  # #     command_list += []

  # # hjb.run_command_list(command_list)
  # # jm.launch_menu(command=command)
  # jm.launch_exp_list(command='echo 2 -e <exp_id>', reset=1, in_parallel=False)
  
  # assert(os.path.exists(os.path.join(savedir_base, hu.hash_dict(exp_list[0]), 'job_dict.json')))
  # summary_list = jm.get_summary_list()
  # print(hr.filter_list(summary_list, {'job_state':'SUCCEEDED'}))
  # print(hr.group_list(summary_list, key='job_state', return_count=True))
  
  # rm = hr.ResultManager(exp_list=exp_list, savedir_base=savedir_base)
  # rm_summary_list = rm.get_job_summary()

  # db = hj.get_dashboard(rm,  wide_display=True)
  # db.display()
  # # assert(rm_summary_list['table'].equals(jm_summary_list['table']))
  
  # # jm.kill_jobs()
  # # assert('CANCELLED' in jm.get_summary()['status'][0])
