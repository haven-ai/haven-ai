from .. import haven_utils as hu
from .. import haven_chk as hc
import os

import time
import copy
import pandas as pd
import numpy as np
import getpass
import pprint
import requests

try:
    import eai_toolkit_client
    from eai_toolkit_client.rest import ApiException
except Exception:
    pass


def to_dict(job):
    return {"id": job.id, "state": job.state, "runs": job.runs, "alive": job.alive}


# Api
# ==============


def get_api(**kwargs):
    # Get Borgy API
    jobs_url = "https://console.elementai.com"
    config = eai_toolkit_client.Configuration()
    config.host = jobs_url

    api_client = eai_toolkit_client.ApiClient(config)

    if kwargs.get("token") is None:
        try:
            token_url = "https://internal.console.elementai.com/v1/token"
            r = requests.get(token_url)
            r.raise_for_status()
            token = r.text
        except requests.exceptions.HTTPError as errh:
            # Perhaps do something for each error
            token = hu.subprocess_call("eai login token -H").split(" ")[-1].replace("\n", "")

        except requests.exceptions.ConnectionError as errc:
            raise SystemExit(errc)
        except requests.exceptions.Timeout as errt:
            raise SystemExit(errt)
        except requests.exceptions.RequestException as err:
            raise SystemExit(err)

    api_client.set_default_header("Authorization", "Bearer {}".format(token))

    # create an instance of the API class
    api = eai_toolkit_client.JobApi(api_client)

    return api


# Job submission
# ==============


def submit_job_v2(api, account_id, command, job_config, savedir):
    workdir = os.path.join(savedir, "code")
    return submit_job(api, account_id, command, job_config, workdir, savedir_logs=savedir)


def submit_job(api, account_id, command, job_config, workdir, savedir_logs=None):
    job_spec = get_job_spec(job_config, command, savedir_logs, workdir=workdir)
    job = api.v1_account_job_post(account_id=account_id, human=1, job_spec=job_spec)
    job_id = job.id
    return job_id


def get_job_spec(job_config, command, savedir_logs, workdir):
    _job_config = copy.deepcopy(job_config)
    _job_config["workdir"] = workdir

    if savedir_logs is not None:
        path_log = os.path.join(savedir_logs, "logs.txt")
        path_err = os.path.join(savedir_logs, "err.txt")
        command_with_logs = "%s 1>%s 2>%s" % (command, path_log, path_err)
    else:
        command_with_logs = command

    _job_config["command"] = ["/bin/bash", "-c", command_with_logs]
    _job_config["resources"] = eai_toolkit_client.JobSpecResources(**_job_config["resources"])
    job_spec = eai_toolkit_client.JobSpec(**_job_config)

    # Return the Job command in Byte format
    return job_spec


# Job status
# ===========
def get_jobs_dict(api, job_id_list, query_size=20):
    # get jobs
    "id__in=64c29dc7-b030-4cb0-8c51-031db029b276,52329dc7-b030-4cb0-8c51-031db029b276"

    jobs = []
    for i in range(0, len(job_id_list), query_size):
        job_id_string = "id__in="
        for job_id in job_id_list[i : i + query_size]:
            job_id_string += "%s," % job_id
        job_id_string = job_id_string[:-1]
        jobs += api.v1_cluster_job_get(q=job_id_string).items

    jobs_dict = {job.id: to_dict(job) for job in jobs}

    return jobs_dict


def get_job(api, job_id):
    """Get job information."""
    try:
        return to_dict(api.v1_job_get_by_id(job_id))
    except ApiException as e:
        raise ValueError("job id %s not found." % job_id)


def get_jobs(api, account_id):
    # account_id = hu.subprocess_call('eai account get').split('\n')[-2].split(' ')[0]
    return [
        to_dict(j)
        for j in api.v1_account_job_get(
            account_id=account_id, limit=1000, order="-created", q="alive_recently=True"
        ).items
    ]

    # return api.v1_me_job_get(limit=1000,
    #         order='-created',
    #         q="alive_recently=True").items


# Job kill
# ===========
def kill_job(api, job_id):
    """Kill a job job until it is dead."""
    job = get_job(api, job_id)

    if not job["alive"]:
        print("%s is already dead" % job_id)
    else:
        # toolkit
        api.v1_job_delete_by_id(job_id)
        print("%s CANCELLING..." % job_id)
        job = get_job(api, job_id)
        while job["state"] == "CANCELLING":
            time.sleep(2.0)
            job = get_job(api, job_id)

        print("%s now is dead." % job_id)
