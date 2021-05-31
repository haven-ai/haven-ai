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

        # hardcoding the savedir in container to /root/results
        tokens = str.split(command)[2:]
        sb_idx = tokens.index('-sb' if '-sb' in tokens else '--savedir_base')
        tokens[sb_idx + 1] = 'results'
        attributes = " ".join(tokens)

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


def setup_image(job_config, savedir_base, exp_list):
    print("Setting up the docker image...")
    try:
        # create docker image
        generate_docker_image = "docker build -q '%s'" % (os.getcwd())
        image_id = hu.subprocess_call(generate_docker_image).strip()

        # copy exp_dict.json files to the image
        for exp_dict in exp_list:
            savedir = os.path.join(savedir_base, hu.hash_dict(exp_dict))
            fname_exp_dict = os.path.join(savedir, "exp_dict.json")
            hu.save_json(fname_exp_dict, exp_dict)
        create_container = "docker create %s" % (image_id)
        container_id = hu.subprocess_call(create_container).strip()

        copy_to_container = "docker cp %s/. %s:/root/results" % (savedir_base, container_id)
        hu.subprocess_call(copy_to_container)

        commit_container = "docker commit %s" % (container_id)
        new_image_id = hu.subprocess_call(commit_container).strip()

        # tag the image with registry path
        registry_path = "%s/%s/%s:%s" % (job_config["container_hostname"], job_config["project_id"],
                                         job_config["project_id"], time.strftime("%Y%m%d_%H%M%S"))
        tag_image = "docker tag '%s' '%s'" % (new_image_id, registry_path)
        hu.subprocess_call(tag_image)

        # uppload the image to the container registry
        upload_iamge = "docker push '%s'" % (registry_path)
        hu.subprocess_call(upload_iamge)

        # remove the images and container
        delete_container = "docker rm %s" % (container_id)
        hu.subprocess_call(delete_container)
        delete_image = "docker rmi -f %s" % (new_image_id)
        hu.subprocess_call(delete_image)

        # add the image name to job_config
        job_config["container_tag"] = registry_path
        return job_config

    except SubprocessError as e:
        raise SystemExit(e)
