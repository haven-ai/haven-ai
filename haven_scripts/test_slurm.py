import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

from haven.haven_jobs import slurm_manager as sm

if __name__ == "__main__":
  job_id = sm.submit_job()
  job = sm.get_job()
  status = sm.kill_job()
