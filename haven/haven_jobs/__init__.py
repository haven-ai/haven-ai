import os
import time
import sys
import subprocess
from .. import haven_utils as hu
from .. import haven_chk as hc
from textwrap import wrap

import copy
import pandas as pd
import numpy as np
import getpass
import pprint

ALIVE_STATES = ["RUNNING", "QUEUED", "PENDING", "QUEUING"]
COMPLETED_STATES = ["COMPLETED", "SUCCEEDED", "COMPLETING"]
FAILED_STATES = ["FAILED", "CANCELLED", "INTERRUPTED", "TIMEOUT", "NODE_FAIL"]


class JobManager:
    """Job manager."""

    def __init__(
        self,
        exp_list=None,
        savedir_base=None,
        workdir=None,
        job_config=None,
        verbose=1,
        account_id=None,
        job_scheduler="toolkit",
        save_logs=True,
    ):
        """[summary]

        Parameters
        ----------
        exp_list : [type]
            [description]
        savedir_base : [type]
            [description]
        workdir : [type], optional
            [description], by default None
        job_config : [type], optional
            [description], by default None
        verbose : int, optional
            [description], by default 1
        """
        if account_id is None and job_config is not None and "account_id" in job_config:
            account_id = job_config["account_id"]
            del job_config["account_id"]

        self.exp_list = exp_list
        self.job_config = job_config
        self.workdir = workdir
        self.verbose = verbose
        self.savedir_base = savedir_base
        self.account_id = account_id
        self.save_logs = save_logs

        # define funcs
        if job_scheduler == "toolkit":
            from . import toolkit_manager as ho

            self.ho = ho
            self.api = self.ho.get_api(token=None)

        elif job_scheduler == "slurm":
            from . import slurm_manager as ho

            self.ho = ho
            self.api = None

    def get_command_history(self, topk=10):
        job_list = self.get_jobs()

        count = 0
        for j in job_list:
            if hasattr(j, "command"):
                print(count, ":", j.command[2])
            if count > topk:
                break
            count += 1

    # Base functions
    # --------------
    def get_jobs(self):
        return self.ho.get_jobs(self.api, account_id=self.account_id)

    def get_jobs_dict(self, job_id_list):
        return self.ho.get_jobs_dict(self.api, job_id_list)

    def get_job(self, job_id):
        return self.ho.get_job(self.api, job_id)

    def kill_job(self, job_id):
        return self.ho.kill_job(self.api, job_id)

    def submit_job(self, command, workdir, savedir_logs=None):
        return self.ho.submit_job(
            api=self.api,
            account_id=self.account_id,
            command=command,
            job_config=self.job_config,
            workdir=workdir,
            savedir_logs=savedir_logs,
        )

    # Main functions
    # --------------
    def launch_menu(self, command=None, exp_list=None, get_logs=False, wait_seconds=3, in_parallel=True):
        exp_list = exp_list or self.exp_list
        summary_list = self.get_summary_list(get_logs=False, exp_list=exp_list)
        summary_dict = hu.group_list(summary_list, key="job_state", return_count=True)

        print("\nTotal Experiments:", len(exp_list))
        print("Experiment Status:", summary_dict)
        prompt = (
            "\nMenu:\n"
            "  0)'ipdb' run ipdb for an interactive session; or\n"
            "  1)'reset' to reset the experiments; or\n"
            "  2)'run' to run the remaining experiments and retry the failed ones; or\n"
            "  3)'status' to view the job status; or\n"
            "  4)'kill' to kill the jobs.\n"
            "Type option: "
        )

        option = input(prompt)

        option_list = ["reset", "run", "status", "logs", "kill"]
        if option not in option_list:
            raise ValueError("Prompt input has to be one of these choices %s" % option_list)

        if option == "ipdb":
            import ipdb

            ipdb.set_trace()
            print("Example:\nsummary_dict = self.get_summary(get_logs=True, exp_list=exp_list)")

        elif option == "status":
            # view experiments
            self.print_job_status(exp_list=exp_list)
            return

        elif option == "reset":
            self.verbose = False
            for e_list in chunk_list(exp_list, n=100):
                self.launch_exp_list(command=command, exp_list=e_list, reset=1, in_parallel=in_parallel)

        elif option == "run":
            self.verbose = False
            tmp_list = [
                s_dict["exp_dict"]
                for s_dict in summary_list
                if s_dict["job_state"] not in ALIVE_STATES + COMPLETED_STATES
            ]

            print("Selected %d/%d exps" % (len(tmp_list), len(exp_list)))
            exp_list = tmp_list
            if len(exp_list) == 0:
                print("All experiments have ran.")
                return

            for e_list in chunk_list(exp_list, n=100):
                self.launch_exp_list(command=command, exp_list=e_list, reset=0, in_parallel=in_parallel)

        elif option == "kill":
            self.verbose = False
            tmp_list = [s_dict["exp_dict"] for s_dict in summary_list if s_dict["job_state"] in ALIVE_STATES]
            self.kill_jobs(exp_list=tmp_list)

        # view experiments
        # print("Checking job status...")
        time.sleep(wait_seconds)
        self.print_job_status(exp_list=exp_list)

    def print_job_status(self, exp_list):
        summary_list = self.get_summary_list(get_logs=False, exp_list=exp_list)
        summary_dict = hu.group_list(summary_list, key="job_state", return_count=False)

        for k in summary_dict.keys():
            n_jobs = len(summary_dict[k])
            if n_jobs:
                # print('\nExperiments %s: %d' % (k, n_jobs))
                pass
                # print(pd.DataFrame(summary_dict[k]).head())

        summary_dict = hu.group_list(summary_list, key="job_state", return_count=True)
        print(summary_dict)

    def launch_exp_list(self, command, exp_list=None, savedir_base=None, reset=0, in_parallel=True):
        exp_list = exp_list or self.exp_list
        assert "<exp_id>" in command

        submit_dict = {}

        if in_parallel:
            pr = hu.Parallel()

            for exp_dict in exp_list:
                exp_id = hu.hash_dict(exp_dict)

                savedir_base = savedir_base or self.savedir_base
                savedir = os.path.join(savedir_base, hu.hash_dict(exp_dict))

                com = command.replace("<exp_id>", exp_id)
                pr.add(self.launch_or_ignore_exp_dict, exp_dict, com, reset, savedir, submit_dict)

            pr.run()
            pr.close()

        else:
            for exp_dict in exp_list:
                exp_id = hu.hash_dict(exp_dict)

                savedir_base = savedir_base or self.savedir_base
                savedir = os.path.join(savedir_base, hu.hash_dict(exp_dict))

                com = command.replace("<exp_id>", exp_id)
                self.launch_or_ignore_exp_dict(exp_dict, com, reset, savedir, submit_dict)

        if len(submit_dict) == 0:
            raise ValueError("The threads have an error, most likely a permission error (see above)")

        for i, (k, v) in enumerate(submit_dict.items()):
            print("***")
            print("Exp %d/%d - %s" % (i + 1, len(submit_dict), v["message"]))
            print("exp_id: %s" % hu.hash_dict(v["exp_dict"]))
            print("job_id: %s" % k)
            savedir = os.path.join(savedir_base, hu.hash_dict(v["exp_dict"]))
            print(f"savedir: {savedir}")
            pprint.pprint(v["exp_dict"])
            print()

        print("%d experiments submitted." % len(exp_list))
        if len(submit_dict) > 0:
            assert len(submit_dict) == len(exp_list), "considered exps does not match expected exps"
        return submit_dict

    def kill_jobs(self, exp_list=None):
        exp_list = exp_list or self.exp_list
        hu.check_duplicates(exp_list)

        pr = hu.Parallel()
        submit_dict = {}

        for exp_dict in exp_list:
            exp_id = hu.hash_dict(exp_dict)
            savedir = os.path.join(self.savedir_base, exp_id)
            fname = get_job_fname(savedir)

            if os.path.exists(fname):
                job_id = hu.load_json(fname)["job_id"]
                pr.add(self.kill_job, job_id)
                submit_dict[exp_id] = "KILLED"
            else:
                submit_dict[exp_id] = "NoN-Existent"

        pr.run()
        pr.close()
        pprint.pprint(submit_dict)
        print("%d/%d experiments killed." % (len([s for s in submit_dict.values() if "KILLED" in s]), len(submit_dict)))
        return submit_dict

    def launch_or_ignore_exp_dict(self, exp_dict, command, reset, savedir, submit_dict={}):
        """launch or ignore job.

        It checks if the experiment exist and manages the special casses, e.g.,
        new experiment, reset, failed, job is already running, completed
        """
        # Define paths
        fname = get_job_fname(savedir)

        if not os.path.exists(fname):
            # Check if the job already exists
            job_dict = self.launch_exp_dict(exp_dict, savedir, command, job=None)
            job_id = job_dict["job_id"]
            message = "SUBMITTED: Launching"

        elif reset:
            # Check if the job already exists
            job_id = hu.load_json(fname).get("job_id")
            self.kill_job(job_id)
            hc.delete_and_backup_experiment(savedir)

            job_dict = self.launch_exp_dict(exp_dict, savedir, command, job=None)
            job_id = job_dict["job_id"]
            message = "SUBMITTED: Resetting"

        else:
            job_id = hu.load_json(fname).get("job_id")
            job = self.get_job(job_id)

            if job["state"] in ALIVE_STATES + COMPLETED_STATES:
                # If the job is alive, do nothing
                message = "IGNORED: Job %s" % job["state"]

            elif job["state"] in FAILED_STATES:
                message = "SUBMITTED: Retrying %s Job" % job["state"]
                job_dict = self.launch_exp_dict(exp_dict, savedir, command, job=job)
                job_id = job_dict["job_id"]
            # This shouldn't happen
            else:
                raise ValueError("wtf")

        submit_dict[job_id] = {"exp_dict": exp_dict, "message": message}

    def launch_exp_dict(self, exp_dict, savedir, command, job=None):
        """Submit a job job and save job dict and exp_dict."""
        # Check for duplicates
        # if job is not None:
        # assert self._assert_no_duplicates(job)

        fname_exp_dict = os.path.join(savedir, "exp_dict.json")
        hu.save_json(fname_exp_dict, exp_dict)
        exp_id = hu.hash_dict(exp_dict)

        assert hu.hash_dict(hu.load_json(fname_exp_dict)) == exp_id

        # Define paths
        workdir_job = os.path.join(savedir, "code")

        # Copy the experiment code into the experiment folder
        print(f"Copying code for experiment {exp_id}")
        hu.copy_code(self.workdir + "/", workdir_job, verbose=0)

        # Run  command
        if self.save_logs:
            savedir_logs = savedir
        else:
            savedir_logs = None
        job_id = self.submit_job(command, workdir_job, savedir_logs=savedir_logs)
        print(f"Job submitted for experiment {exp_id} with job id {job_id}")

        # Verbose
        if self.verbose:
            print("Job_id: %s command: %s" % (job_id, command))

        job_dict = {"job_id": job_id, "command": command}

        hu.save_json(get_job_fname(savedir), job_dict)

        return job_dict

    def get_summary_list(
        self,
        failed_only=False,
        columns=None,
        max_lines=10,
        wrap_size=8,
        add_prefix=False,
        get_logs=True,
        exp_list=None,
        savedir_base=None,
    ):
        savedir_base = savedir_base or self.savedir_base
        exp_list = exp_list or self.exp_list

        # get job key
        job_id_list = []
        for exp_dict in exp_list:
            exp_id = hu.hash_dict(exp_dict)
            savedir = os.path.join(savedir_base, exp_id)
            fname = get_job_fname(savedir)

            if os.path.exists(fname):
                job_id_list += [hu.load_json(fname)["job_id"]]

        jobs_dict = self.get_jobs_dict(job_id_list)

        # get summaries
        summary_list = []

        for exp_dict in exp_list:
            result_dict = {}

            exp_id = hu.hash_dict(exp_dict)
            savedir = os.path.join(savedir_base, exp_id)
            job_fname = get_job_fname(savedir)

            # General info
            result_dict = {}
            result_dict["exp_dict"] = exp_dict
            result_dict["exp_id"] = exp_id
            result_dict["job_id"] = None
            result_dict["job_state"] = "NEVER LAUNCHED"

            if os.path.exists(job_fname):
                job_dict = hu.load_json(job_fname)
                job_id = job_dict["job_id"]
                if job_id not in jobs_dict:
                    continue

                fname_exp_dict = os.path.join(savedir, "exp_dict.json")
                job = jobs_dict[job_id]

                # if hasattr(job, 'command'):
                #     command = job_dict['command']
                # else:
                #     command = None

                # Job info
                result_dict["started_at"] = hu.time_to_montreal(fname_exp_dict)
                result_dict["job_id"] = job_id
                result_dict["job_state"] = job["state"]
                result_dict["restarts"] = len(job["runs"])
                result_dict["command"] = job_dict["command"]

                if get_logs:
                    # Logs info
                    if job["state"] == "FAILED":
                        logs_fname = os.path.join(savedir, "err.txt")
                    else:
                        logs_fname = os.path.join(savedir, "logs.txt")

                    if os.path.exists(logs_fname):
                        result_dict["logs"] = hu.read_text(logs_fname)[-max_lines:]

            summary_list += [result_dict]

        return summary_list

    def get_summary(self, **kwargs):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        # get job ids

        # fill summary
        summary_dict = {"table": [], "status": [], "logs_failed": [], "logs": []}

        # get info
        df = pd.DataFrame(summary_dict["table"])

        # if columns:
        #     df = df[[c for c in columns if (c in df.columns and c not in ['err'])]]

        if "job_state" in df:
            stats = np.vstack(np.unique(df["job_state"].fillna("NaN"), return_counts=True)).T
            status = [{a: b} for (a, b) in stats]
        else:
            df["job_state"] = None

        df = hu.sort_df_columns(df)
        summary_dict["status"] = status
        summary_dict["table"] = df

        for state in ALIVE_STATES + COMPLETED_STATES + FAILED_STATES:
            summary_dict[state] = df[df["job_state"] == state]

        return summary_dict

    def _assert_no_duplicates(self, job_new=None, max_jobs=500):
        # Get the job list
        jobList = self.get_jobs()

        # Check if duplicates already exist in job
        command_dict = {}
        for job in jobList:

            if hasattr(job, "command"):
                if job.command is None:
                    continue
                job_python_command = job.command[2]
            else:
                job_python_command = None

            if job_python_command is None:
                continue
            elif job_python_command not in command_dict:
                command_dict[job_python_command] = job
            else:
                print("Job state", job["state"], "Job command", job_python_command)
                raise ValueError("Job %s is duplicated" % job_python_command)

        # Check if the new job causes duplicate
        if job_new is not None:
            if job_new.command[2] in command_dict:
                job_old_id = command_dict[job_new.command[2]].id
                raise ValueError("Job exists as %s" % job_old_id)

        return True


def get_job_fname(savedir):
    if os.path.exists(os.path.join(savedir, "borgy_dict.json")):
        # for backward compatibility
        fname = os.path.join(savedir, "borgy_dict.json")
    else:
        fname = os.path.join(savedir, "job_dict.json")

    return fname


def chunk_list(my_list, n=100):
    return [my_list[x : x + n] for x in range(0, len(my_list), n)]
