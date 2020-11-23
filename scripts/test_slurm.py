import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

from haven.haven_jobs import slurm_manager as sm

if __name__ == "__main__":
  # return
  exp_list = [{'model':{'name':'mlp', 'n_layers':20}, 
              'dataset':'mnist', 'batch_size':1}]
  savedir_base = '.tmp'

  
  jm = hjb.JobManager(exp_list=exp_list, 
                  savedir_base=savedir_base, 
                  workdir=os.path.dirname(os.path.realpath(__file__)),
                  job_config=job_config,
                  )
  # get jobs              
  job_list_old = jm.get_jobs()

  # run single command
  savedir_logs = '%s/%s' % (savedir_base, np.random.randint(1000))
  os.makedirs(savedir_logs, exist_ok=True)
  command = 'echo 2'
  job_id = jm.submit_job(command,  workdir=jm.workdir, savedir_logs=savedir_logs)

  # get jobs
  job_list = jm.get_jobs()
  job = jm.get_job(job_id)
  assert job_list[0].id == job_id
  
  # jm.kill_job(job_list[0].id)
  # run
  print('jobs:', len(job_list_old), len(job_list))
  assert (len(job_list_old) + 1) ==  len(job_list)

  # command_list = []
  # for exp_dict in exp_list:
  #     command_list += []

  # hjb.run_command_list(command_list)
  # jm.launch_menu(command=command)
  jm.launch_exp_list(command='echo 2 -e <exp_id>', reset=1, in_parallel=False)
  
  assert(os.path.exists(os.path.join(savedir_base, hu.hash_dict(exp_list[0]), 'job_dict.json')))
  summary_list = jm.get_summary_list()
  print(hr.filter_list(summary_list, {'job_state':'SUCCEEDED'}))
  print(hr.group_list(summary_list, key='job_state', return_count=True))
  
  rm = hr.ResultManager(exp_list=exp_list, savedir_base=savedir_base)
  rm_summary_list = rm.get_job_summary()

  db = hj.get_dashboard(rm,  wide_display=True)
  db.display()
  # assert(rm_summary_list['table'].equals(jm_summary_list['table']))
  
  # jm.kill_jobs()
  # assert('CANCELLED' in jm.get_summary()['status'][0])
