from .. import haven_utils as hu
from .. import haven_chk as hc

import time
import os
from subprocess import SubprocessError
import json

def submit_job(api, account_id, command, job_config, workdir, savedir_logs=None):

    # todo: fix the data stuff!! cp to image, done in image prep
    try:
        job_name = "%s_%s" % (job_config["project_id"], time.strftime("%Y%m%d_%H%M%S"))
        job_name = ''.join(list(map(lambda c: c if c.isalnum() else '_', job_name)))

        # the savedir in container is always /root/results
        tokens = str.split(command)[2:]
        sb_idx = tokens.index('-sb' if '-sb' in tokens else '--savedir_base')
        tokens[sb_idx + 1] = 'results'
        haven_attr = " ".join(tokens)
        haven_attr += " -gc gs://%s" % (job_config["gcloud_savedir"])

        gcp_attr = ""
        for key in list(job_config.keys()):
            # skip gcp_manager attr
            if key in ["account_id", "gcloud_savedir", "container_hostname", "project_id"]:
                continue
            gcp_attr += " --%s %s" % (key, job_config[key])

        submit_job_command = "gcloud ai-platform jobs submit training %s %s -- %s" % (
            job_name, gcp_attr, haven_attr)

        hu.subprocess_call(submit_job_command)
        return job_name

    except SubprocessError as e:
        raise SystemExit(e.output.decode('utf-8'))


def get_job(api, job_id):
    job_info = get_jobs_dict(api, [job_id])[job_id]
    job_info['job_id'] = job_id
    return job_info


def get_jobs(api, account_id):
    # use the default logged in account
    command = "gcloud ai-platform jobs list --format='json(jobId,state)'"
    job_info = hu.subprocess_call(command)
    job_info = json.loads(job_info)
    result = []
    for i in job_info:
        i["job_id"] = i.pop("jobId")
        i["runs"] = []
        result.append(i)
    return result


def get_jobs_dict(api, job_id_list, query_size=20):
    job_id_list = ",".join(job_id_list)
    command = "gcloud ai-platform jobs list --filter='JOB_ID=(%s)' --format='json(jobId,state)' --limit='%s'" % (
        job_id_list, str(query_size))
    job_info = hu.subprocess_call(command)
    job_info = json.loads(job_info)
    temp = {}
    for i in job_info:
        job_id = i.pop("jobId")
        i["runs"] = []
        temp[job_id] = i
    job_info = temp
    return job_info


def kill_job(api, job_id):
    """Kill a job job until it is dead."""
    job = get_job(api, job_id)

    if job["state"] in ["CANCELLED", "CANCELLING", "SUCCEEDED", "FAILED"]:
        print("%s is already dead" % job_id)
    else:
        kill_command = "gcloud ai-platform jobs cancel %s" % (job_id)
        while True:
            try:
                hu.subprocess_call(kill_command)
                print("%s CANCELLING..." % job_id)
            except Exception as e:
                # todo: not properly handled
                print(str(e.output))
            break

        # confirm cancelled
        job = get_job(api, job_id)
        if job["state"] == "CANCELLED":
            print("%s now is dead." % job_id)


def setup_image(job_config, savedir_base, exp_list):
    print("Setting up the docker image...")
    try:
        # create docker image
        generate_docker_image = "docker build -q '%s'" % (os.getcwd())
        image_id = hu.subprocess_call(generate_docker_image).strip()

        # copy exp_dict.json files to the image
        temp_folder = os.path.join(savedir_base, 'temp')
        for exp_dict in exp_list:
            savedir = os.path.join(temp_folder, hu.hash_dict(exp_dict))
            fname_exp_dict = os.path.join(savedir, "exp_dict.json")
            hu.save_json(fname_exp_dict, exp_dict)
        create_container = "docker create %s" % (image_id)
        container_id = hu.subprocess_call(create_container).strip()

        copy_to_container = "docker cp %s/. %s:/root/results" % (temp_folder, container_id)
        hu.subprocess_call(copy_to_container)

        commit_container = "docker commit %s" % (container_id)
        new_image_id = hu.subprocess_call(commit_container).strip()

        # tag the image with registry path
        registry_path = "%s/%s/%s:%s" % (job_config["container_hostname"], job_config["project_id"],
                                         job_config["project_id"], time.strftime("%Y%m%d_%H%M%S"))
        tag_image = "docker tag '%s' '%s'" % (new_image_id, registry_path)
        hu.subprocess_call(tag_image)

        print("Pushing the docker image to google cloud...")
        # uppload the image to the container registry
        upload_iamge = "docker push '%s'" % (registry_path)
        hu.subprocess_call(upload_iamge)

        # remove the images and container
        delete_container = "docker rm %s" % (container_id)
        hu.subprocess_call(delete_container)
        delete_image = "docker rmi -f %s" % (new_image_id)
        hu.subprocess_call(delete_image)
        delete_temp_folder = "rm -rf %s" % (temp_folder)
        hu.subprocess_call(delete_temp_folder)

        # add the image name to job_config
        job_config["master-image-uri"] = registry_path
        return job_config

    except SubprocessError as e:
        raise SystemExit(e.output.decode('utf-8'))


def download_results(exp_list, savedir_base, gcloud_base):
    str_exp_list = ('|').join(map(lambda d: hu.hash_dict(d), exp_list))
    sync_command = "gsutil -m rsync -r -x '(?!%s).*/.*\.(?!pkl$).*' %s %s" % (str_exp_list, gcloud_base, savedir_base)
    try:
        hu.subprocess_call(sync_command)
    except SubprocessError as e:
        print(e.output.decode('utf-8'))


def download_all(exp_list, savedir_base, gcloud_base):
    str_exp_list = ('|').join(map(lambda d: hu.hash_dict(d), exp_list))
    sync_command = "gsutil -m rsync -r -x (?!%s) %s %s" % (str_exp_list, gcloud_base, savedir_base)
    try:
        hu.subprocess_call(sync_command)
    except SubprocessError as e:
        print(e.output.decode('utf-8'))


def download_log(job_id, serverity):
    # todo: low performance
    command = 'gcloud logging read "resource.labels.job_id=%s AND severity=%s" --format="json(jsonPayload.message)"' % (job_id, serverity)
    log = hu.subprocess_call(command)
    log = json.loads(log)
    return ''.join([s for s in [l['jsonPayload']['message'] for l in log if l] if s])
    
