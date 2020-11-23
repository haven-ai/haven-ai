from .. import haven_utils as hu
from .. import haven_chk as hc
import os

import time
import copy
import pandas as pd
import numpy as np 
import getpass
import pprint
import requests


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
