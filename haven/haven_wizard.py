import os
import sys

import argparse
import haven

from . import haven_utils as hu


def get_args():
    parser = argparse.ArgumentParser(formatter_class=make_wide(argparse.ArgumentDefaultsHelpFormatter))

    parser.add_argument("-e", "--exp_group_list", nargs="+", help="Define which exp groups to run.")
    parser.add_argument(
        "-sb", "--savedir_base", default=None, help="Define the base directory where the experiments will be saved."
    )
    parser.add_argument("-r", "--reset", default=0, type=int, help="Reset or resume the experiment.")
    parser.add_argument("--exp_id", default=None, help="Run a specific experiment based on its id.")
    parser.add_argument(
        "-j", "--job_scheduler", default=None, type=str, help="Run the experiments as jobs in the cluster."
    )

    args, others = parser.parse_known_args()

    return args


def make_wide(formatter, w=120, h=36):
    """Return a wider HelpFormatter, if possible."""
    try:
        # https://stackoverflow.com/a/5464440
        # beware: "Only the name of this class is considered a public API."
        kwargs = {"width": w, "max_help_position": h}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        import warnings

        warnings.warn("argparse help formatter failed, falling back.")
        return formatter


def run_wizard(
    func,
    exp_list=None,
    exp_groups=None,
    job_config=None,
    savedir_base=None,
    reset=None,
    args=None,
    use_threads=False,
    exp_id=None,
    python_binary_path="python",
    python_file_path=None,
    workdir=None,
    job_scheduler=None,
    save_logs=True,
    filter_duplicates=False,
    results_fname=None,
    job_option=None,
    job_copy_ignore_patterns=None,
    job_ignore_status=None,
):
    """
    Runs a set of experiments either locally or on a cluster.

    It does the following:
        - creates a unique id for each experiment
        - creates a folder for each experiment
        - copies the code for each experiment
        - saves the hyperparameters for each experiment
        - runs the experiment
    """
    if args is None:
        args = get_args()
        custom_args = {}
    else:
        custom_args = vars(args).copy()
        for k, v in vars(get_args()).items():
            if k in custom_args:
                continue
            setattr(args, k, v)

    # Asserts
    # =======
    savedir_base = savedir_base or args.savedir_base
    reset = reset or args.reset
    exp_id = exp_id or args.exp_id
    assert savedir_base is not None

    # make sure savedir_base is absolute
    savedir_base = os.path.abspath(savedir_base)

    # Collect experiments
    # ===================
    if exp_id is not None:
        # select one experiment
        savedir = os.path.join(savedir_base, exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    elif exp_list is None:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_groups[exp_group_name]

    if filter_duplicates:
        n_total = len(exp_list)
        exp_list = hu.filter_duplicates(exp_list)
        print(f"Filtered {len(exp_list)}/{n_total}")

    hu.check_duplicates(exp_list)
    print("\nRunning %d experiments" % len(exp_list))

    # save results folder
    if exp_id is None and results_fname is not None:
        if len(results_fname):
            if ".ipynb" not in results_fname:
                raise ValueError(".ipynb should be the file extension")
            hu.create_jupyter_file(fname=results_fname, savedir_base=savedir_base)

    # Run experiments
    # ===============
    if job_scheduler is None:
        job_scheduler = args.job_scheduler

    if job_scheduler in [None, "None", "0"]:
        job_scheduler = None

    elif job_scheduler in ["toolkit", "slurm", "gcp"]:
        job_scheduler = job_scheduler

    elif job_scheduler in ["1"]:
        job_scheduler = "toolkit"

    else:
        raise ValueError(f"{job_scheduler} does not exist")

    if job_scheduler is None:
        for exp_dict in exp_list:
            savedir = create_experiment(exp_dict, savedir_base, reset=reset, verbose=True)
            # do trainval
            func(exp_dict=exp_dict, savedir=savedir, args=args)

    else:
        # launch jobs
        print(f"Using Job Scheduler: {job_scheduler}")

        from haven import haven_jobs as hjb

        assert job_config is not None
        assert "account_id" in job_config

        if workdir is None:
            workdir = os.getcwd()

        jm = hjb.JobManager(
            exp_list=exp_list,
            savedir_base=savedir_base,
            workdir=workdir,
            job_config=job_config,
            job_scheduler=job_scheduler,
            save_logs=save_logs,
            job_copy_ignore_patterns=job_copy_ignore_patterns,
            job_ignore_status=job_ignore_status,
        )

        if python_file_path is None:
            python_file_path = os.path.split(sys.argv[0])[-1]

        command = f"{python_binary_path} {python_file_path} --exp_id <exp_id> --savedir_base {savedir_base} --python_binary '{python_binary_path}'"

        for k, v in custom_args.items():
            if k not in [
                "python_binary",
                "savedir_base",
                "sb",
                "exp_id",
                "e",
                "exp_group_list",
                "j",
                "job_scheduler",
                "r",
                "reset",
            ]:
                command += f" --{k} {v}"

        print(command)
        jm.launch_menu(command=command, in_parallel=use_threads, job_option=job_option)


def create_experiment(exp_dict, savedir_base, reset, copy_code=False, return_exp_id=False, verbose=True):
    import pprint
    from . import haven_chk as hc

    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        hc.delete_and_backup_experiment(savedir)

    # create experiment structure
    os.makedirs(savedir, exist_ok=True)

    # -- save exp_dict only when it is needed
    exp_dict_json_fname = os.path.join(savedir, "exp_dict.json")
    if not os.path.exists(exp_dict_json_fname):
        hu.save_json(exp_dict_json_fname, exp_dict)
    else:
        # make sure it is not corrupt and same exp_id
        try:
            exp_dict_tmp = hu.load_json(exp_dict_json_fname)
            assert hu.hash_dict(exp_dict_tmp) == hu.hash_dict(exp_dict)
        except:
            hu.save_json(exp_dict_json_fname, exp_dict)

    # -- images
    os.makedirs(os.path.join(savedir, "images"), exist_ok=True)

    if copy_code:
        src = os.getcwd() + "/"
        dst = os.path.join(savedir, "code")
        hu.copy_code(src, dst)

    if verbose:
        print("\n******")
        print(f"Haven: {haven.__version__}")
        print("Exp id: %s" % exp_id)
        print("\nHyperparameters:\n" + "-" * 16)
        # print(pd.DataFrame([exp_dict]).to_string(index=False))
        pprint.pprint(exp_dict)

        print("\nSave directory: %s" % savedir)
        print("=" * 100)

    if return_exp_id:
        return savedir, exp_id

    return savedir
