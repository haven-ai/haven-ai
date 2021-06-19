"""
  Minimal code for launching commands on cluster
"""
import os

from haven import haven_jobs as hjb


if __name__ == "__main__":
    # Choose Job Scheduler
    job_scheduler = "toolkit"
    if job_scheduler == "slurm":
        job_config = {
            "account_id": "def-dnowrouz-ab",
            "time": "1:00:00",
            "cpus-per-task": "2",
            "mem-per-cpu": "20G",
            "gres": "gpu:1",
        }

    elif job_scheduler == "toolkit":
        import job_configs

        job_config = job_configs.JOB_CONFIG

    savedir_base = os.path.abspath("../results")

    # run the trainval
    command = f"python trainval.py -sb {savedir_base} -e syn -r 1"

    # This command copies a snapshot of the code, saves the logs and errors, keeps track of the job status, keeps backup, and ensures one unique command per job
    job = hjb.launch_job(
        command, savedir_base=savedir_base, job_scheduler=job_scheduler, job_config=job_config, reset=True
    )
