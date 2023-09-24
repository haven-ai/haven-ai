import contextlib
import tqdm
import copy
import hashlib
import json
import pprint
import os
import shlex
import subprocess
import threading
import time
import numpy as np
import itertools
import pandas as pd
import pickle
import shutil

try:
    import nbformat as nbf
except:
    pass

from datetime import datetime


def create_command(base_command, args):
    """
    args is the parser
    """
    run_command = base_command
    arg_keys = vars(args).keys()

    assert "exp_group_list" in arg_keys
    assert "exp_id" in arg_keys
    assert "run_jobs" in arg_keys

    for a, v in vars(args).items():
        if a == "exp_group_list" or a == "exp_id" or a == "run_jobs" or a == "reset":
            print("argument: %s ignored..." % a)
            continue

        run_command += " --%s %s" % (a, v)
    print("command: %s" % run_command)
    return run_command


def load_txt(fname):
    """Load the content of a txt file.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    list
        Content of the file. List containing the lines of the file
    """
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines


def save_txt(fname, lines):
    """Load the content of a txt file.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    list
        Content of the file. List containing the lines of the file
    """
    # turn fname into a string in
    fname = str(fname)

    with open(fname, "w", encoding="utf-8") as f:
        for l in lines:
            f.writelines(l)


def torch_load(fname, map_location=None):
    """Load the content of a torch file.

    Parameters
    ----------
    fname : str
        File name
    map_location : [type], optional
        Maping the loaded model to a specific device (i.e., CPU or GPU), this
        is needed if trained in CPU and loaded in GPU and viceversa, by default
        None

    Returns
    -------
    [type]
        Loaded torch model
    """
    import torch

    obj = torch.load(fname, map_location=map_location)

    return obj


def torch_save(fname, obj):
    """Save data in torch format.

    Parameters
    ----------
    fname : str
        File name
    obj : [type]
        Data to save
    """
    import torch

    # Create folder
    os.makedirs(os.path.dirname(fname), exist_ok=True)  # TODO: add makedirs parameter?

    # Define names of temporal files
    fname_tmp = fname + ".tmp"  # TODO: Make the safe flag?

    torch.save(obj, fname_tmp)
    if os.path.exists(fname):
        os.remove(fname)
    os.rename(fname_tmp, fname)


class Parallel:
    """Class for run a function in parallel."""

    def __init__(self):
        """Constructor"""
        self.threadList = []
        self.count = 0
        self.thread_logs = []

    def add(self, func, *args):
        """Add a function to run as a process.

        Parameters
        ----------
        func : function
            Pointer to the function to parallelize
        args : list
            Arguments of the funtion to parallelize
        """
        self.threadList += [threading.Thread(target=func, name="thread-%d" % self.count, args=args)]
        self.count += 1

    def run(self):
        """Run the added functions in parallel"""
        for thread in tqdm.tqdm(self.threadList, desc="Starting threads", leave=False):
            thread.daemon = True
            thread.start()

    def close(self):
        """Finish: wait for all the functions to finish"""
        for thread in tqdm.tqdm(self.threadList, desc="Joining threads", leave=False):
            thread.join()


def pretty_print_df(df):
    # wrap text for prettiness
    for c in df.columns:
        if df[c].dtype == "O":
            # df[c] = df[c].str.wrap(wrap_size)
            df[c] = df[c].apply(pprint.pformat)
    return df


def flatten_column(result_dict, flatten_list=False):
    new_dict = {}

    for k, v in result_dict.items():
        if isinstance(v, list) and flatten_list:
            list_dict = {}
            for vi in v:
                flat = flatten_dict(k, vi)
                for f1, f2 in flat.items():
                    if f1 in list_dict:
                        list_dict[f1] += "_" + str(f2)
                    else:
                        list_dict[f1] = str(f2)
            # print()
        else:
            list_dict = flatten_dict(k, v)

        new_dict.update(list_dict)

    result_dict = new_dict
    return result_dict


def sort_df_columns(table, also_first=[]):
    first = ["exp_id", "job_state", "job_id", "restarts", "started_at"]
    first += also_first
    col_list = []
    for col in first:
        if col in table.columns:
            col_list += [col]
    for col in table.columns:
        if col in first:
            continue
        col_list += [col]

    return table[col_list]


def subprocess_call(cmd_string):
    """Run a terminal process.

    Parameters
    ----------
    cmd_string : str
        Command to execute in the terminal

    Returns
    -------
    [type]
        Error code or 0 if no error happened
    """
    return subprocess.check_output(shlex.split(cmd_string), shell=False, stderr=subprocess.STDOUT).decode("utf-8")


def copy_code(src_path, dst_path, verbose=1, use_rsync=True, ignore_patterns=None):
    """Copy the code.

    Typically, when you run an experiment, first you copy the code used to the
    experiment folder. This function copies the code using rsync terminal
    command.

    Parameters
    ----------
    src_path : str
        Source code directory
    dst_path : str
        Destination code directory
    verbose : int, optional
        Verbosity level. If 0 does not print stuff, by default 1

    Raises
    ------
    ValueError
        [description]
    """
    if verbose:
        print("\nCopying Code\n  - src: %s\n  - dst: %s\n" % (src_path, dst_path))

    # Create destination folder
    os.makedirs(dst_path, exist_ok=True)

    # Check if rsync is available
    rsync_avialable = len(subprocess.run(["which", "rsync"], capture_output=True, text=True).stdout) > 0

    if rsync_avialable and use_rsync:
        # Define the command for copying the code using rsync
        if os.path.exists(os.path.join(src_path, ".havenignore")):
            copy_code_cmd = (
                "rsync -av -r -q  --delete-before --exclude='.*'  --exclude-from=%s \
                            --exclude '__pycache__/' %s %s"
                % (os.path.join(src_path, ".havenignore"), src_path, dst_path)
            )
        else:
            copy_code_cmd = (
                "rsync -av -r -q  --delete-before --exclude='.*' \
                            --exclude '__pycache__/' %s %s"
                % (src_path, dst_path)
            )
        # Run the command in the terminal
        try:
            subprocess_call(copy_code_cmd)
        except subprocess.CalledProcessError as e:
            raise ValueError("Ping stdout output:\n", e.output)

    else:
        # delete the destination folder
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)

        # load from havenignore
        ignore = None
        if ignore_patterns is None:
            if os.path.exists(os.path.join(src_path, ".havenignore")):
                # read from .havenignore like from .gitignore
                ignore_patterns = []
                with open(os.path.join(src_path, ".havenignore"), "r") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line.startswith("#") or line == "":
                            continue
                        ignore_patterns += [line]
                #
                ignore = shutil.ignore_patterns(*ignore_patterns)
        else:
            ignore = shutil.ignore_patterns(*ignore_patterns)

        # copy the code folder
        shutil.copytree(src_path, dst_path, ignore=ignore)


def zipdir(src_dirname, out_fname, include_list=None):
    """Compress a folder using ZIP.

    Parameters
    ----------
    src_dirname : str
        Directory to compress
    out_fname : str
        File name of the compressed file
    include_list : list, optional
        List of files to include. If None, include all files in the folder, by
        default None
    """
    import zipfile  # TODO: Move to the beggining of the file

    # TODO: Do we need makedirs?
    # Create the zip file
    zipf = zipfile.ZipFile(out_fname, "w", zipfile.ZIP_DEFLATED)

    # ziph is zipfile handle
    for root, dirs, files in os.walk(src_dirname):
        for file in files:
            # Descard files if needed
            if include_list is not None and file not in include_list:
                continue

            abs_path = os.path.join(root, file)
            rel_path = fname_parent(abs_path)  # TODO: fname_parent not defined
            print(rel_path)
            zipf.write(abs_path, rel_path)

    zipf.close()


def zip_score_list(exp_list, savedir_base, out_fname, include_list=None):
    """Compress a list of experiments in zip.

    Parameters
    ----------
    exp_list : list
        List of experiments to zip
    savedir_base : str
        Directory where the experiments from the list are saved
    out_fname : str
        File name for the zip file
    include_list : list, optional
        List of files to include. If None, include all files in the folder, by
        default None
    """
    for exp_dict in exp_list:  # TODO: This will zip only the last experiments, zipdir will overwritwe the previous file
        # Get the experiment id
        exp_id = hash_dict(exp_dict)
        # Zip folder
        zipdir(os.path.join(savedir_base, exp_id), out_fname, include_list=include_list)


def time_to_montreal(fname=None, timezone="US/Eastern"):
    """Get time in Montreal zone.

    Returns
    -------
    str
        Current date at the selected timezone in string format
    """
    # Get time
    os.environ["TZ"] = timezone
    try:
        time.tzset()
    except:
        pass
    if fname:
        tstamp = os.path.getctime(fname)
    else:
        tstamp = time.time()

    time_str = datetime.fromtimestamp(tstamp).strftime("%I:%M %p (%b %d)")

    return time_str


@contextlib.contextmanager
def random_seed(seed):
    """[summary]

    Parameters
    ----------
    seed : [type]
        [description]
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def is_subset(d1, d2, strict=False):
    """[summary]

    Parameters
    ----------
    d1 : [type]
        [description]
    d2 : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    flag = True
    for k in d1:
        v1, v2 = d1.get(k), d2.get(k)

        # if both are values
        if not isinstance(v2, dict) and not isinstance(v1, dict):
            if v1 != v2:
                flag = False

        # if both are dicts
        elif isinstance(v2, dict) and isinstance(v1, dict):
            flag = is_subset(v1, v2)

        # if d1 is dict and not d2
        elif isinstance(v1, dict) and not isinstance(v2, dict):
            flag = False

        # if d1 is not and d2 is dict
        elif not isinstance(v1, dict) and isinstance(v2, dict):
            flag = False

        if flag is False:
            break

    return flag


def as_double_list(v):
    """[summary]

    Parameters
    ----------
    v : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if not isinstance(v, list) and not isinstance(v, np.ndarray):
        v = [v]

    if not isinstance(v[0], list) and not isinstance(v[0], np.ndarray):
        v = [v]

    return v


def ignore_duplicates(list_of_dicts):
    # ensure no duplicates in exp_list
    dict_list = []
    hash_list = set()
    for data_dict in list_of_dicts:
        dict_id = hash_dict(data_dict)
        if dict_id in hash_list:
            continue
        else:
            hash_list.add(dict_id)
            dict_list += [data_dict]
    return dict_list


def filter_duplicates(list_of_dicts):
    # ensure no duplicates in exp_list
    tmp_list = []
    hash_list = set()
    for data_dict in list_of_dicts:
        dict_id = hash_dict(data_dict)
        if dict_id in hash_list:
            continue
        else:
            hash_list.add(dict_id)
        tmp_list += [data_dict]

    return tmp_list


def check_duplicates(list_of_dicts):
    # ensure no duplicates in exp_list
    hash_list = set()
    for data_dict in list_of_dicts:
        dict_id = hash_dict(data_dict)
        if dict_id in hash_list:
            raise ValueError("duplicated dictionary detected:\n%s" % pprint.pformat(data_dict))
        else:
            hash_list.add(dict_id)


def load_py(fname):
    """[summary]

    Parameters
    ----------
    fname : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    import sys
    from importlib import reload
    from importlib import import_module

    if not os.path.exists(fname):
        raise ValueError("%s not found..." % fname)

    sys.path.append(os.path.dirname(fname))

    name = os.path.split(fname)[-1].replace(".py", "")
    module = import_module(name)
    reload(module)
    sys.path.pop()

    return module


def get_exp_list_from_ids(exp_id_list, savedir_base):
    exp_list = []
    for exp_id in exp_id_list:
        exp_list += [load_json(os.path.join(savedir_base, exp_id, "exp_dict.json"))]
    return exp_list


def flatten_dict(key_name, v_dict):
    if not isinstance(v_dict, dict):
        return {key_name: v_dict}

    leaf_dict = {}
    for k in v_dict:
        if key_name != "":
            k_new = key_name + "." + k
        else:
            k_new = k
        leaf_dict.update(flatten_dict(key_name=k_new, v_dict=v_dict[k]))

    return leaf_dict


def get_diff_hparam(exp_list):
    df = pd.DataFrame([flatten_column(e) for e in exp_list])
    return get_diff_columns(df, min_threshold=2, max_threshold="auto")


def get_diff_columns(df, min_threshold=2, max_threshold="auto"):
    df.reset_index()
    if max_threshold == "auto":
        max_threshold = df.shape[0]

    if max_threshold < 0:
        max_threshold = df.shape[0] + max_threshold

    column_count = []

    for column in df.columns:
        _set = set([str(v) for v in df[column].values])
        column_count.append(len(_set))
    indices = np.arange(len(df.columns))

    column_count = np.array(column_count)

    indices = indices[(column_count >= min_threshold) & (column_count <= max_threshold)]
    diff_columns = [df.columns[i] for i in indices]

    return diff_columns


def timeit(func, n_times=10, **args):
    for i in range(n_times):
        if i == 1:
            s = time.time()
        func(**args)

    print("time:", (time.time() - s) / (n_times - 1))


def cartesian_exp_group(exp_config, remove_none=False):
    """Cartesian experiment config.

    It converts the exp_config into a list of experiment configuration by doing
    the cartesian product of the different configuration. It is equivalent to
    do a grid search.

    Parameters
    ----------
    exp_config : str
        Dictionary with the experiment Configuration

    Returns
    -------
    exp_list: str
        A list of experiments, each defines a single set of hyper-parameters
    """
    exp_config_copy = copy.deepcopy(exp_config)

    # Make sure each value is a list
    for k, v in exp_config_copy.items():
        if not isinstance(exp_config_copy[k], list):
            exp_config_copy[k] = [v]

    # Create the cartesian product
    exp_list_raw = (
        dict(zip(exp_config_copy.keys(), values)) for values in itertools.product(*exp_config_copy.values())
    )

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        # remove hparams with None
        if remove_none:
            to_remove = []
            for k, v in exp_dict.items():
                if v is None:
                    to_remove += [k]
            for k in to_remove:
                del exp_dict[k]
        exp_list += [exp_dict]

    return exp_list


def _filter_fn(val):
    if val == "true":
        return True
    elif val == "false":
        return False
    elif val == "none":
        return None
    else:
        return val


def _traverse(dict_, keys, val):
    if len(keys) == 0:
        return
    else:
        # recurse
        if keys[0] not in dict_:
            if len(keys[1:]) == 0:
                dict_[keys[0]] = _filter_fn(val)
            else:
                dict_[keys[0]] = {}
        _traverse(dict_[keys[0]], keys[1:], val)


def get_exp_list_from_json(fname=None, json_dict=None):
    if fname is not None:
        json_dict = json.loads(open(fname).read())

    exps = cartesian_exp_group(json_dict)
    return [unflatten(dd) for dd in exps]


def unflatten(dict_):
    new_dict = {}
    for key, val in dict_.items():
        key_split = key.split(".")
        _traverse(new_dict, key_split, val)
    return new_dict


def hash_dict(exp_dict):
    """Create a hash for an experiment.

    Parameters
    ----------
    exp_dict : dict
        An experiment, which is a single set of hyper-parameters

    Returns
    -------
    hash_id: str
        A unique id defining the experiment
    """
    dict2hash = ""
    if not isinstance(exp_dict, dict):
        raise ValueError("exp_dict is not a dict")

    for k in sorted(exp_dict.keys()):
        if "." in k:
            raise ValueError(". has special purpose")
        elif isinstance(exp_dict[k], dict):
            v = hash_dict(exp_dict[k])
        elif isinstance(exp_dict[k], tuple):
            raise ValueError(f"{exp_dict[k]} tuples can't be hashed yet, consider converting tuples to lists")
        elif isinstance(exp_dict[k], list) and len(exp_dict[k]) and isinstance(exp_dict[k][0], dict):
            v_str = ""
            for e in exp_dict[k]:
                if isinstance(e, dict):
                    v_str += hash_dict(e)
                else:
                    raise ValueError("all have to be dicts")
            v = v_str
        else:
            v = exp_dict[k]

        dict2hash += str(k) + "/" + str(v)
    hash_id = hashlib.md5(dict2hash.encode()).hexdigest()

    return hash_id


def hash_str(str):
    """Hashes a string using md5.

    Parameters
    ----------
    str
        a string

    Returns
    -------
    str
        md5 hash for the input string
    """
    return hashlib.md5(str.encode()).hexdigest()


def group_exp_list(exp_list, groupby_list):
    """Split the experiment list into smaller lists where each
       is grouped by a set of hyper-parameters

    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    groupby_list : list
        The set of hyper-parameters to group the experiments

    Returns
    -------
    lists_of_exp_list : list
        Experiments grouped by a set of hyper-parameters

    Example
    -------
    >>>
    >>>
    >>>
    """
    if groupby_list is None:
        return [exp_list]
    if not isinstance(groupby_list, list):
        groupby_list = [groupby_list]
    # groupby_list = as_double_list(groupby_list)
    exp_list = copy.deepcopy(exp_list)
    exp_list_flat = [flatten_column(exp_dict, flatten_list=True) for exp_dict in exp_list]
    for e1, e2 in zip(exp_list_flat, exp_list):
        e1["exp_dict"] = e2

    def split_func(x):
        x_list = []
        for k_list in groupby_list:
            if not isinstance(k_list, list):
                k_list = [k_list]
            val = get_str(x, k_list)
            x_list += [val]

        return x_list

    exp_list_flat.sort(key=split_func)

    list_of_exp_list = []
    group_dict = itertools.groupby(exp_list_flat, key=split_func)

    # exp_group_dict = {}
    for k, v in group_dict:
        v_list = [vi["exp_dict"] for vi in list(v)]
        list_of_exp_list += [v_list]
    #     # print(k)
    #     exp_group_dict['_'.join(list(map(str, k)))] = v_list

    return list_of_exp_list


def get_exp_diff(exp_list):
    flat_list = [flatten_column(e, flatten_list=True) for e in exp_list]

    hparam_dict = {}
    for f in flat_list:
        for k, v in f.items():
            if k not in hparam_dict:
                hparam_dict[k] = set()
            hparam_dict[k].add(v)
    count_dict = []
    for k, v in hparam_dict.items():
        count_dict += [{"name": k, "uniques": len(v)}]

    count_dict = sorted(count_dict, key=lambda i: -i["uniques"])
    return count_dict


def group_list(python_list, key, return_count=False):
    group_dict = {}
    for p in python_list:
        p_tmp = copy.deepcopy(p)
        del p_tmp[key]
        k = p[key]

        if k not in group_dict:
            group_dict[k] = []

        group_dict[k] += [p_tmp]

    if return_count:
        count_dict = {}
        for k in group_dict:
            count_dict[k] = len(group_dict[k])
        return count_dict
    return group_dict


def get_exp_list_from_config(exp_groups, exp_config_fname):
    exp_list = []
    for e in exp_groups:
        exp_list += load_py(exp_config_fname).EXP_GROUPS[e]

    return exp_list


def get_str(h_dict, k_list):
    k = k_list[0]

    if len(k_list) == 1:
        return str(h_dict.get(k))

    return get_str(h_dict.get(k), k_list[1:])


def get_best_exp_dict(
    exp_list,
    savedir_base,
    metric,
    metric_agg="min",
    filterby_list=None,
    avg_across=None,
    return_scores=False,
    verbose=True,
    score_list_name="score_list.pkl",
    return_score_dict=False,
):
    """Obtain best the experiment for a specific metric.

    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : [type]
        A directory where experiments are saved
    metric : [type]
        [description]
    min_or_max : [type]
        [description]
    return_scores : bool, optional
        [description], by default False
    """
    scores_dict = []
    if metric_agg in ["min", "min_last"]:
        best_score = np.inf
    elif metric_agg in ["max", "max_last"]:
        best_score = 0.0

    exp_dict_best = None
    best_score_dict = None
    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)
    for exp_dict in exp_list:
        exp_id = hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)

        score_list_fname = os.path.join(savedir, score_list_name)
        if not os.path.exists(score_list_fname):
            if verbose:
                print("%s: missing %s" % (exp_id, score_list_name))
            continue

        score_list = load_pkl(score_list_fname)
        metric_scores = [score_dict[metric] for score_dict in score_list if metric in score_dict]
        if len(metric_scores) == 0:
            continue
        if metric_agg in ["min", "min_last"]:
            if metric_agg == "min_last":
                score = metric_scores[-1]
            elif metric_agg == "min":
                score = np.min(metric_scores)

            if best_score >= score:
                best_score = score
                exp_dict_best = exp_dict
                best_score_dict = {"score": score, "epochs": len(score_list), "exp_id": exp_id}

        elif metric_agg in ["max", "max_last"]:
            if metric_agg == "max_last":
                score = metric_scores[-1]
            elif metric_agg == "max":
                score = np.max(metric_scores)

            if best_score <= score:
                best_score = score
                exp_dict_best = exp_dict
                best_score_dict = {"score": score, "epochs": len(score_list), "exp_id": exp_id}

    if exp_dict_best is None:
        if verbose:
            print('no experiments with metric "%s"' % metric)
        return {}

    if return_score_dict:
        return exp_dict_best, best_score_dict

    return exp_dict_best


def get_exp_list_from_exp_configs(exp_group_list, workdir, filterby_list=None, verbose=True):
    """[summary]

    Parameters
    ----------
    exp_group_list : [type]
        [description]
    workdir : [type]
        [description]
    filterby_list : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    assert workdir is not None

    from importlib import reload

    assert workdir is not None
    if workdir not in sys.path:
        sys.path.append(workdir)
    import exp_configs as ec

    reload(ec)

    exp_list = []
    for exp_group in exp_group_list:
        exp_list += ec.EXP_GROUPS[exp_group]
    if verbose:
        print("%d experiments" % len(exp_list))

    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)
    return exp_list


def get_exp_ids(savedir_base, filterby_list=None, verbose=True):
    return get_exp_list(savedir_base, filterby_list, verbose, return_ids=True)


def get_exp_list(savedir_base, filterby_list=None, verbose=True, return_ids=False):
    """[summary]

    Parameters
    ----------
    savedir_base : [type], optional
        A directory where experiments are saved, by default None
    filterby_list : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    exp_list = []
    dir_list = os.listdir(savedir_base)

    for exp_id in dir_list:
        savedir = os.path.join(savedir_base, exp_id)
        fname = os.path.join(savedir, "exp_dict.json")
        if len(exp_id) != 32:
            # exp_id is not a hash
            continue

        if not os.path.exists(fname):
            # missing exp_dict.json
            continue

        exp_dict = load_json(fname)
        expected_id = hash_dict(exp_dict)
        if expected_id != exp_id:
            if verbose:
                # assert(hash_dict(exp_dict) == exp_id)
                print("%s does not match %s" % (expected_id, exp_id))
            continue

        exp_list += [exp_dict]

    exp_list = filter_exp_list(exp_list, filterby_list)
    if return_ids:
        return [hash_dict(e) for e in exp_list]

    if verbose:
        # number of experiments loaded
        print("Number of experiments loaded: %d " % len(exp_list))
        # print location of experiments
        print(f"Experiments location: {savedir_base}")

    return exp_list


def zip_exp_list(savedir_base):
    """[summary]

    Parameters
    ----------
    savedir_base : [type]
        [description]
    """
    import zipfile

    with zipfile.ZipFile(savedir_base) as z:
        for filename in z.namelist():
            if not os.path.isdir(filename):
                # read the file
                with z.open(filename) as f:
                    for line in f:
                        print(line)


def filter_list(python_list, filterby_list, verbose=True):
    return filter_exp_list(python_list, filterby_list, verbose=verbose)


def filter_exp_list(
    exp_list, filterby_list, savedir_base=None, verbose=True, score_list_name="score_list.pkl", return_style_list=False
):
    """[summary]

    Parameters
    ----------
    exp_list : [type]
        A list of experiments, each defines a single set of hyper-parameters
    filterby_list : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if filterby_list is None or filterby_list == "" or len(filterby_list) == 0:
        if return_style_list:
            return exp_list, [{}] * len(exp_list)
        else:
            return exp_list

    style_list = []
    filterby_list_list = as_double_list(filterby_list)
    # filterby_list = filterby_list_list

    for filterby_list in filterby_list_list:
        exp_list_new = []

        # those with meta
        filterby_list_no_best = []
        for filterby_dict in filterby_list:
            meta_dict = {}
            if isinstance(filterby_dict, tuple):
                fd, meta_dict = filterby_dict

            if meta_dict.get("best"):
                assert savedir_base is not None
                el = filter_exp_list(exp_list, filterby_list=fd, verbose=verbose)
                best_dict = meta_dict.get("best")
                exp_dict = get_best_exp_dict(
                    el,
                    savedir_base,
                    metric=best_dict["metric"],
                    metric_agg=best_dict["metric_agg"],
                    filterby_list=None,
                    avg_across=best_dict.get("avg_across"),
                    return_scores=False,
                    verbose=verbose,
                    score_list_name=score_list_name,
                )

                exp_list_new += [exp_dict]
                style_list += [meta_dict.get("style", {})]
            else:
                filterby_list_no_best += [filterby_dict]

        # ignore metas here meta
        for exp_dict in exp_list:
            select_flag = False

            for fd in filterby_list_no_best:
                if isinstance(fd, tuple):
                    filterby_dict, meta_dict = fd
                    style_dict = meta_dict.get("style", {})
                else:
                    filterby_dict = fd
                    style_dict = {}

                filterby_dict = copy.deepcopy(filterby_dict)

                keys = filterby_dict.keys()

                tmp_dict = {}
                for k in keys:
                    if "." in k:
                        v = filterby_dict[k]
                        k_list = k.split(".")
                        nk = len(k_list)

                        dict_tree = dict()
                        t = dict_tree

                        for i in range(nk):
                            ki = k_list[i]
                            if i == (nk - 1):
                                t = t.setdefault(ki, v)
                            else:
                                t = t.setdefault(ki, {})

                        tmp_dict.update(dict_tree)
                    else:
                        tmp_dict[k] = filterby_dict[k]

                filterby_dict = tmp_dict
                assert isinstance(filterby_dict, dict), "filterby_dict: %s is not a dict" % str(filterby_dict)

                if is_subset(filterby_dict, exp_dict):
                    select_flag = True
                    break

            if select_flag:
                exp_list_new += [exp_dict]
                style_list += [style_dict]

        exp_list = exp_list_new

    if verbose:
        print("Filtered: %d/%d experiments gathered..." % (len(exp_list_new), len(exp_list)))
    # check_duplicates(exp_list_new)
    exp_list_new = ignore_duplicates(exp_list_new)

    if return_style_list:
        return exp_list_new, style_list

    return exp_list_new


import ast


def get_dict_from_str(string):
    if string is None:
        return string

    if string == "None":
        return None

    if string == "":
        return None

    return ast.literal_eval(string)


def get_list_from_str(string):
    if string is None:
        return string

    if string == "None":
        return None

    string = string.replace(" ", "").replace("]", "").replace("[", "").replace('"', "").replace("'", "")

    if string == "":
        return None

    return string.split(",")


def save_json(fname, data, makedirs=True):
    """Save data into a json file.

    Parameters
    ----------
    fname : str
        Name of the json file
    data : [type]
        Data to save into the json file
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    # turn fname to string in case it is a Path object
    fname = str(fname)
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def load_mat(fname):
    """Load a matlab file.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    dict
        Dictionary with the loaded data
    """
    import scipy.io as io

    return io.loadmat(fname)


def load_json(fname, decode=None):  # TODO: decode???
    """Load a json file.

    Parameters
    ----------
    fname : str
        Name of the file
    decode : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        Content of the file
    """
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d


def read_text(fname):
    """Loads the content of a text file.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    list
        Content of the file. List containing the lines of the file
    """
    with open(fname, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return lines


def load_pkl(fname):
    """Load the content of a pkl file.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    [type]
        Content of the file
    """
    with open(fname, "rb") as f:
        return pickle.load(f)


def save_pkl(fname, data, with_rename=True, makedirs=True):
    """Save data in pkl format.

    Parameters
    ----------
    fname : str
        File name
    data : [type]
        Data to save in the file
    with_rename : bool, optional
        [description], by default True
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    # turn fname to string in case it is a Path object
    fname = str(fname)

    # create folder
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)

    # Save file
    if with_rename:
        fname_tmp = fname + "_tmp.pth"
        with open(fname_tmp, "wb") as f:
            pickle.dump(data, f)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(fname_tmp, fname)
    else:
        with open(fname, "wb") as f:
            pickle.dump(data, f)


def get_exp_list_from_exp_configs(exp_configs_fname, exp_group):
    # get experiments from the exp config
    config = load_py(exp_configs_fname)

    # return exp_group
    return config.EXP_GROUPS[exp_group]


# Jupyter notebook utils
# ==============================================================================
def create_jupyter_file(
    fname,
    savedir_base,
    overwrite=False,
):
    """
    Create a jupyter notebook file to visualize the results.
    """
    savedir_base = os.path.abspath(savedir_base)
    if overwrite or not os.path.exists(fname):
        nb = nbf.v4.new_notebook()

        nb["cells"] = [
            main_markdown(),
            load_exps_md(),
            load_exps_cell(savedir_base),
            load_results_md(),
            load_results_cell(),
            load_jobs_md(),
            load_jobs_cell(),
            load_plots_md(),
            load_plots_cell(),
            install_cell(),
        ]

        if os.path.dirname(fname) != "":
            os.makedirs(os.path.dirname(fname), exist_ok=True)

        with open(fname, "w") as f:
            nbf.write(nb, f)

        print("> Open %s to visualize results" % fname)


def main_markdown():
    script = """
    # Visualization Scripts for your experiments

In this notebook, you will find various visualization scripts to analyze the results of your experiments.


    # Table of Contents
* [1. Load Experiments](#load-experiments)
* [2. Visualize Results Table](#visualize-results-table)
* [3. Check Job Status and Logs](#check-job-status-and-logs)
* [4. Generate Plots](#generate-plots)
"""

    return nbf.v4.new_markdown_cell(script)


# Experiments Section
# ==============================================================================


def load_exps_md():
    script = """### 1. Load Experiments
----------------------

- The following cell will load selected experiments from a given directory called `savedir_base`.
- It returns 'rm' which is the result manager object that will be used to visualize the results

#### Instructions:
- Change 'savedir_base' to the directory where your experiments are saved.
- if you want to load all experiments in `savedir_base`, set `load_all=True`.
- if you want to load specific experiments from `exp_configs.py`, set `load_all=False`.
    - replace `<path_to_exp_configs.py>` with your `exp_configs.py` absolute path
    - replace `<which_exp_group>` with the name of the experiment group you want to load
- Note that experiments without score_list.pkl will not be plotted as they contain the metric scores.
          """
    return nbf.v4.new_markdown_cell(script)


def load_exps_cell(savedir_base):
    script = (
        """
'''
1. Load Experiments
'''
import pandas as pd
import os 

from haven import haven_utils as hu
from haven import haven_results as hr

def load_experiments():
    # 1. Change Path to where the results are saved
    savedir_base = "%s"

    # 2. Define which experiments to load
    load_all = True

    if load_all:
        # Load all experiments in savedir_base
        exp_list = None
    else:
        # Load specific experiments from exp_configs.py
        exp_configs_fname = "<path_to_exp_configs.py>" 
        exp_group = "<which_exp_group>" 
        exp_list = hu.get_exp_list_from_exp_configs(exp_configs_fname, exp_group)

    # 3. Load experiments
    rm = hr.ResultManager(exp_list=exp_list,
                        savedir_base=savedir_base,
                        filterby_list=None,
                        verbose=1,
                        job_scheduler='toolkit'
                        )
    return rm

rm = load_experiments()
          """
        % savedir_base
    )
    return nbf.v4.new_code_cell(script)


# 2. Table of Results Section
# ==============================================================================


def load_results_md():
    script = """### 2. Load Results Table
----------------------

- The following cell will display the table of results.
- It returns 'df' a dataframe with the results

#### Instructions:
- Modify `hparam_columns` to display the hyperparameters you want.
    -  Print `rm.hparam_columns` to find out which hyperparameters are available.
- Modify `score_columns` to display the metrics you want.
    - Print `rm.score_columns` to find out which metrics are available.
          """
    return nbf.v4.new_markdown_cell(script)


def load_results_cell():
    script = """'''
2. Load Table of Results
'''
# Reload Experiments
rm = load_experiments()

# print hparam_columns score_columns
print("Available Hyperparameters:\\n", rm.hparam_columns)
print("Available Score Columns:\\n",rm.score_columns)

# define hyperparameters and metrics to display
hparam_columns = None # could be for instance ['n_layers']
score_columns = None # could be for instance ['loss', 'acc']

# get table
df = rm.get_score_df(columns=hparam_columns, score_columns=score_columns, 
                     show_max_min=False, show_exp_ids=True)

display(df)
          """
    return nbf.v4.new_code_cell(script)


# 3. Table of Jobs Section
# ==============================================================================
def load_jobs_md():
    script = """### 3. Load Jobs Table
----------------------

- The following cell will display the table of job status and logs.
- It returns 'df' a dataframe with the job status and logs

#### Instructions:
- Define `exp_id` to print its logs
          """
    return nbf.v4.new_markdown_cell(script)


def load_jobs_cell():
    script = """'''
3. Load Table of Job Status and Logs
'''
# Reload Experiments
rm = load_experiments()

# Increase width of dataframes
pd.set_option('display.max_colwidth', 1000)

# Get the right columns
columns = ['exp_id', 'job_state', 'job_id', 'logs']
df = pd.DataFrame(rm.get_job_summary())
df = df[[c for c in columns if c in df.columns]]

# add savedir by combining savedir_base and exp_id
df['savedir'] = df['exp_id'].apply(lambda x: os.path.join(rm.savedir_base, x))

# for each unique job_state show the table
for job_state in df['job_state'].unique():
    print(f"Job State: {job_state}\\n============")
    display(df[df['job_state'] == job_state].head())

# print the count of each job_state
print("Job State Counts\\n================")
print(df['job_state'].value_counts())

# print the logs of a specific exp_id
exp_id = None
if exp_id is not None:
    print(f"Logs of exp_id {exp_id}\\n, df[df['exp_id'] == exp_id]['logs']")
          """
    return nbf.v4.new_code_cell(script)


# 4. Plots Section
# ==============================================================================
def load_plots_md():
    script = """### 4. Plots Results
----------------------

- The following cell will display plots of the results
- It returns matplotlib figures

#### Instructions:
- Modify `legend_list` to display the hyperparameters you want.
    -  Print `rm.hparam_columns` to find out which hyperparameters are available.
- Modify `y_metric_list` and 'x_metric' to plot the metrics you want.
    - Print `rm.score_columns` to find out which metrics are available.
          """
    return nbf.v4.new_markdown_cell(script)


def load_plots_cell():
    script = """'''
4. Plot Results
'''
# Reload Experiments
rm = load_experiments()

# Define plot arguments
mode = 'line'          # 'line' or 'bar' plots

# print hparam_columns score_columns
print("Available Hyperparameters:\\n", rm.hparam_columns)
print("Available Score Columns:\\n",rm.score_columns)

# Choose from the hparam columns for the legend
legend_list=['model'] # put the hyperparameters you want to plot on the legend here

# Choose from the score columns for (x,y)
y_metric_list=['loss'] # put the metrics you want to plot on the y-axis here
x_metric='epoch'       # put the metric you want to plot on the x-axis here

fig = rm.get_plot_all(
                y_metric_list=y_metric_list, 
                x_metric=x_metric,
                legend_list=legend_list,
                mode=mode
)
          """
    return nbf.v4.new_code_cell(script)


def install_cell():
    script = """!pip install --upgrade git+https://github.com/haven-ai/haven-ai
          """
    return nbf.v4.new_code_cell(script)


def save_ipynb(fname, script_list):
    # turn fname to string in case it is a Path object
    fname = str(fname)

    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_code_cell(code) for code in script_list]
    with open(fname, "w") as f:
        nbf.write(nb, f)
