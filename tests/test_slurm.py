
import unittest
import numpy as np 
import os, sys
import torch
import shutil, time
import copy

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jobs as hjb
from haven import haven_jupyter as hj

if __name__ == '__main__':
    # step 1 - run a job through slurm using `hu.subprocess_call`
    job_id = hu.subprocess_call(submit_command)
    
    # step 2 - get the status of the job from the job_id
    job_status = hu.subprocess_call(get_command)
    
    # step 3 - kill the job
    info = hu.subprocess_call(kill_command)
    
