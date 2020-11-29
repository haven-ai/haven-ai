import os
# need to set up the environment variable before use
ACCOUNT_ID = os.environ['SLURM_ACCOUNT']

# change the output directory
JOB_CONFIG = {
    'time': '12:00:00',
    'cpus-per-task': '2',
    'mem-per-cpu': '20G',
    'gres': 'gpu:p100:1'
}
