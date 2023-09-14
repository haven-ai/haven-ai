from haven import haven_jobs as hjb
from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import unittest
import numpy as np
import os
import sys
import torch
import shutil
import time
import copy
import argparse

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)


def create_slurm_batch(account, time, mem_cpu, command, file_name):
    # create a slurm script with user input
    lines = ("#! /bin/bash \n" "#SBATCH --account=%s \n" "#SBATCH --time=%s \n" "#SBATCH --mem-per-cpu=%s \n" "%s") % (
        account,
        time,
        mem_cpu,
        command,
    )
    hu.save_txt(file_name, lines)


if __name__ == "__main__":

    # specify the slurm script to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    submit_command = "sbatch " + args.batch
    # step 1 - run a job through slurm using `hu.subprocess_call`
    job_id = hu.subprocess_call(submit_command).split()[-1]

    # step 2 - get the status of the job from the job_id
    get_command = "squeue --job %s" % job_id
    job_status = hu.subprocess_call(get_command)

    # step 3 - kill the job
    kill_command = "scancel %s" % job_id
    info = hu.subprocess_call(kill_command)  # no return message after scancel
