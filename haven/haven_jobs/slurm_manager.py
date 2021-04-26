from .. import haven_utils as hu
from .. import haven_chk as hc
import os

import time
import copy
import pandas as pd
import numpy as np
import getpass
import pandas as pd
from subprocess import SubprocessError

# Job submission
# ==============


def submit_job(api, account_id, command, job_config, workdir, savedir_logs=None):
    # read slurm setting
    lines = "#! /bin/bash \n"
    lines += "#SBATCH --account=%s \n" % account_id
    for key in list(job_config.keys()):
        lines += "#SBATCH --%s=%s \n" % (key, job_config[key])
    path_log = os.path.join(savedir_logs, "logs.txt")
    lines += "#SBATCH --output=%s \n" % path_log
    path_err = os.path.join(savedir_logs, "err.txt")
    lines += "#SBATCH --error=%s \n" % path_err
    path_code = os.path.join(savedir_logs, "code")
    lines += "#SBATCH --chdir=%s \n" % path_code

    lines += command

    file_name = os.path.join(savedir_logs, "bash.sh")
    hu.save_txt(file_name, lines)

    # launch the exp
    submit_command = "sbatch %s" % file_name
    while True:
        try:
            job_id = hu.subprocess_call(submit_command).split()[-1]
        except SubprocessError as e:
            if "Socket timed out" in str(e.output):
                print("sbatch time out and retry now")
                time.sleep(1)
                continue
            else:
                # other errors
                exit(str(e.output)[2:-1].replace("\\n", ""))
        break

    # delete the bash.sh
    os.remove(file_name)

    return job_id


# Job status
# ===========
def get_job(api, job_id):
    """Get job information."""
    job_info = get_jobs_dict(None, [job_id])[job_id]
    job_info["job_id"] = job_id

    return job_info


def get_jobs(api, account_id):
    """ get all jobs launched by the current user"""
    job_list = ""
    command = 'squeue --user=%s --format="%%.18i %%.8T"' % getpass.getuser()
    while True:
        try:
            job_list = hu.subprocess_call(command)
        except SubprocessError as e:
            if "Socket timed out" in str(e.output):
                print("squeue time out and retry now")
                time.sleep(1)
                continue
            else:
                # other errors
                exit(str(e.output)[2:-1].replace("\\n", ""))
        break

    result = [{"job_id": j.split()[0], "state": j.split()[1]} for j in job_list.split("\n")[1:-1]]
    return result


def get_jobs_dict(api, job_id_list, query_size=20):
    if len(job_id_list) == 0:
        return {}

    jobs_dict = {}

    command = "sacct --jobs=%s --format=jobid,cputime,state" % str(job_id_list)[1:-1].replace(" ", "")
    while True:
        try:
            job_list = hu.subprocess_call(command)
        except SubprocessError as e:
            if "Socket timed out" in str(e.output):
                print("sacct time out and retry now")
                time.sleep(1)
                continue
            else:
                # other errors
                exit(str(e.output)[2:-1].replace("\\n", ""))
        break

    lines = job_list.split("\n")
    header = lines[0].split()
    lines = [l.split() for l in lines[2:-1]]

    df = pd.DataFrame(data=lines, columns=header)
    df = df[~df["JobID"].str.contains(r"\.")]
    df = df.rename(mapper={"State": "state", "CPUTime": "cpuTime", "JobID": "job_id"}, axis=1)
    df = df.replace({"state": r"CANCELLED.*"}, {"state": "CANCELLED"}, regex=True)
    df.insert(loc=0, column="runs", value="")

    # use job id as key
    new_df = df.drop(labels="job_id", axis=1)
    new_df.index = df["job_id"].to_list()
    jobs_dict = new_df.to_dict(orient="index")

    return jobs_dict


# Job kill
# ===========


def kill_job(api, job_id):
    """Kill a job job until it is dead."""
    job = get_job(api, job_id)

    if job["state"] in ["CANCELLED", "COMPLETED", "FAILED", "TIMEOUT"]:
        print("%s is already dead" % job_id)
    else:
        kill_command = "scancel %s" % job_id
        while True:
            try:
                hu.subprocess_call(kill_command)
                print("%s CANCELLING..." % job_id)
            except Exception as e:
                if "Socket timed out" in str(e):
                    print("scancel time out and retry now")
                    time.sleep(1)
                    continue
            break

        # confirm cancelled
        job = get_job(api, job_id)
        while job["state"] != "CANCELLED":
            time.sleep(2)
            job = get_job(api, job_id)

        print("%s now is dead." % job_id)
