from .. import haven_utils as hu
from .. import haven_chk as hc

import time
import os
from subprocess import SubprocessError


def submit_job(api, account_id, command, job_config, workdir, savedir_logs=None):
    # assume Dockerfile exist, actually all jobs share the same image, no need to build so many time..
    # todo: handle exception

    # todo: probably need to fix the account_id and current project things...
    # todo: fix the data stuff!! may need to fetch data from the cloud
    try:
        # todo: better choice of job_name?
        job_name = "%s_%s" % (job_config["project_id"], time.strftime("%Y%m%d_%H%M%S"))
        job_name = ''.join(list(map(lambda c: c if c.isalnum() else '_', job_name)))

        # todo: fix the save directory in container
        tokens = str.split(command)[2:]
        sb_idx = tokens.index('-sb' if '-sb' in tokens else '--savedir_base')
        tokens[sb_idx+1] = 'results'
        attributes = " ".join(tokens)

        # attributes = " ".join(str.split(command)[2:])
        submit_job_command = "gcloud ai-platform jobs submit training %s --region %s --master-image-uri %s -- %s -gc %s" % (
            job_name, job_config["region"], job_config["container_tag"], attributes, job_config["gcloud_savedir"])

        hu.subprocess_call(submit_job_command)
        return job_name

    except SubprocessError as e:
        # todo: fix the output?
        raise SystemExit(e.output) 


def get_job(api, job_id):
    # todo: handle exception
    command = "gcloud ai-platform jobs describe %s" % (job_id)
    job_info = hu.subprocess_call(command)
    return job_info


def get_jobs_dict(api, job_id_list, query_size=20):
    # todo: use --filter and --format for easier string processing
    command = "gcloud ai-platform jobs list"
    job_info = hu.subprocess_call(command)
    pass


def kill_job(api, job_id):
    pass


def setup_image(job_config):
    print("Setting up the docker image...")
    try:
        creation_time = time.strftime("%Y%m%d_%H%M%S")
        image_name = "%s:%s" % (job_config["project_id"], creation_time)
        generate_docker_image = "docker build -t '%s' '%s'" % (image_name, os.getcwd())
        hu.subprocess_call(generate_docker_image)

        # modify this to make it use tag
        container_tag = "%s/%s/%s:%s" % (job_config["container_hostname"], job_config["project_id"], job_config["project_id"], creation_time)
        tag_image = "docker tag '%s' '%s'" % (image_name, container_tag)
        hu.subprocess_call(tag_image)

        # uppload the image to the container registry
        upload_iamge = "docker push '%s'" % (container_tag)    
        hu.subprocess_call(upload_iamge)

        # add the image name to job_config
        job_config["container_tag"] = container_tag
        return job_config

    except SubprocessError as e:
        raise SystemExit(e)
