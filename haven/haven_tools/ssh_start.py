import os, subprocess, time
import argparse, pprint

 
JOB_CONFIG = """
image: registry.console.elementai.com/{ACCOUNT}/ssh
data:
    - {ACCOUNT}.home:/mnt/home
    - {ACCOUNT}.results:/mnt/results
    - {ACCOUNT}.datasets:/mnt/datasets
    - {ACCOUNT}.public:/mnt/public
    - {ACCOUNT}.private:/mnt/private
resources:
    cpu: 8
    mem: 8
    gpu: 1
interactive: true
command:
    - /tk/bin/start.sh
    - /bin/bash
    - -c
    - jupyter notebook --notebook-dir='/mnt' --ip=0.0.0.0 --port=8080 --no-browser --NotebookApp.token='' --NotebookApp.custom_display_url=https://${EAI_JOB_ID}.job.console.elementai.com --NotebookApp.disable_check_xsrf=True --NotebookApp.allow_origin='*'
        """

def get_interactive_job_id(account):
    command = ('eai job ls --account %s --fields state '
                    '--fields id,interactive | '
                    'grep RUNNING | grep true' % account)
    print(command)
    out = os.popen(command).read()
    print('out: %s' % out)

    if out == '':
        return None 

    job_id = out.split()[1]
    return job_id

def kill_job(job_id):
    os.popen('eai job kill %s ' % job_id).read()

def wait_until_state(job_id, required_state):
    elapsed = 0
    max_elapsed = 60

    state = get_job_state(job_id)
    while state != required_state and elapsed < max_elapsed:
        print('%s interactive job: %s (Elapsed %s/%s)' % 
             (state, job_id, elapsed, max_elapsed))
        time.sleep(2)
        state = get_job_state(job_id)
        elapsed += 2

def launch_interactive_job(role):
    command = ('eai job submit --no-header '
               '-f job.yml '
               '--role %s' % (role))

    print(command)
    out = os.popen(command).read()
    print('out: %s' % out)

    return out.split(' ')[0]

def get_job_state(job_id):
    command = ('eai job  info %s '  
                 '| grep state:' % job_id)
    state = os.popen(command).read()
    return state.split(':')[-1].replace('\n','').strip(' ')

def do_port_forwarding(job_id):
    out = os.popen('kill $(lsof -ti:2222)').read()
    command = ('eai job port-forward %s 2222' % (job_id))

    print(command)
    out = os.popen(command).read()
    print('out: %s' % out)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # example user: 'eai.issam'
    parser.add_argument('-a', '--account', required=True)
    # example role: 'eai.issam.admin'
    parser.add_argument('-r', '--role', required=True)
    args = parser.parse_args()

    # define job.yml
    JOB_CONFIG = JOB_CONFIG.replace('{ACCOUNT}', args.account)
    pprint.pprint(JOB_CONFIG)
    print('role:', args.role)
    with open('job.yml', 'w') as outfile:
        outfile.write(JOB_CONFIG)

    # make sure no interactive job is running
    job_id = get_interactive_job_id(args.account)
    if job_id == None or job_id == '':
        out = 'y'
    else:
        # interactive job exists
        print('RUNNING interactive job:', job_id)
        out = input('Restart? (y/n) ')
        if out == 'y':
            # kill job until dead
            kill_job(job_id)
            wait_until_state(job_id, required_state='CANCELLED')

    if out == 'y':
        # run new interactive job until it runs
        job_id = launch_interactive_job(role=args.role)
        if job_id == None or job_id == '':
            print('No job was launched')
        else:
            wait_until_state(job_id, required_state='RUNNING')

        # launch ssh
        # do_port_forwarding(job_id)

    # help commands
    print()
    print('Helpers:\n========\n')
    print('eai job port-forward %s 2222' % job_id)
    print('workstation: ssh -p 2222 toolkit@localhost -v')
    print('jupyter url: https://%s.job.console.elementai.com/' % job_id)

    if os.path.exists('job.yml'):
        os.remove('job.yml')
