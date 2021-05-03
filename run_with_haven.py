"""
  Minimal code for launching commands on cluster
"""

from haven import haven_jobs as hjb

SLURM_JOB_CONFIG = {
    "account_id": "def-dnowrouz-ab",
    "time": "1:00:00",
    "cpus-per-task": "2",
    "mem-per-cpu": "20G",
    "gres": "gpu:1",
}

if __name__ == "__main__":
    # run the trainval
    command = "python trainval.py -e syn -r 1"
    
    # This command copies a snapshot of the code, saves the logs and errors, keeps track of the job status, keeps backup, and ensures one unique command per job
    job = hjb.submit_job(command, savedir_base='../results', job_scheduler='slurm', job_config=SLURM_JOB_CONFIG, reset=True)
    print(job)
