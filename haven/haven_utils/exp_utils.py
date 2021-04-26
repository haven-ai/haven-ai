import os
import json
import copy
import sys
import hashlib
import itertools
from itertools import groupby
import tqdm
import pprint
import numpy as np
from .. import haven_utils as hu


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
            raise ValueError("tuples can't be hashed yet, consider converting tuples to lists")
        elif isinstance(exp_dict[k], list) and isinstance(exp_dict[k][0], dict):
            v_str = ""
            for e in exp_dict[k]:
                if isinstance(e, dict):
                    v_str += hash_dict(e)
                else:
                    raise ValueError("all have to be dicts")
            v = v_str
        else:
            v = exp_dict[k]

        dict2hash += os.path.join(str(k), str(v))

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
    # groupby_list = hu.as_double_list(groupby_list)
    exp_list = copy.deepcopy(exp_list)
    exp_list_flat = [hu.flatten_column(exp_dict) for exp_dict in exp_list]
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
    group_dict = groupby(exp_list_flat, key=split_func)

    # exp_group_dict = {}
    for k, v in group_dict:
        v_list = [vi["exp_dict"] for vi in list(v)]
        list_of_exp_list += [v_list]
    #     # print(k)
    #     exp_group_dict['_'.join(list(map(str, k)))] = v_list

    return list_of_exp_list


def get_exp_diff(exp_list):
    flat_list = [hu.flatten_column(e) for e in exp_list]

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
        exp_list += hu.load_py(exp_config_fname).EXP_GROUPS[e]

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
    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)

        score_list_fname = os.path.join(savedir, score_list_name)
        if not os.path.exists(score_list_fname):
            if verbose:
                print("%s: missing %s" % (exp_id, score_list_name))
            continue

        score_list = hu.load_pkl(score_list_fname)
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

        elif metric_agg in ["max", "max_last"]:
            if metric_agg == "max_last":
                score = metric_scores[-1]
            elif metric_agg == "max":
                score = np.max(metric_scores)

            if best_score <= score:
                best_score = score
                exp_dict_best = exp_dict

        scores_dict += [{"score": score, "epochs": len(score_list), "exp_id": exp_id}]

    if exp_dict_best is None:
        if verbose:
            print('no experiments with metric "%s"' % metric)
        return {}

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

    for exp_id in tqdm.tqdm(dir_list):
        savedir = os.path.join(savedir_base, exp_id)
        fname = os.path.join(savedir, "exp_dict.json")
        if len(exp_id) != 32:
            if verbose:
                print('"%s/" is not an exp directory' % exp_id)
            continue

        if not os.path.exists(fname):
            if verbose:
                print("%s: missing exp_dict.json" % exp_id)
            continue

        exp_dict = hu.load_json(fname)
        expected_id = hu.hash_dict(exp_dict)
        if expected_id != exp_id:
            if verbose:
                # assert(hu.hash_dict(exp_dict) == exp_id)
                print("%s does not match %s" % (expected_id, exp_id))
            continue
        # print(hu.hash_dict(exp_dict))
        exp_list += [exp_dict]

    exp_list = filter_exp_list(exp_list, filterby_list)
    if return_ids:
        return [hu.hash_dict(e) for e in exp_list]
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
    filterby_list_list = hu.as_double_list(filterby_list)
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

                if hu.is_subset(filterby_dict, exp_dict):
                    select_flag = True
                    break

            if select_flag:
                exp_list_new += [exp_dict]
                style_list += [style_dict]

        exp_list = exp_list_new

    if verbose:
        print("Filtered: %d/%d experiments gathered..." % (len(exp_list_new), len(exp_list)))
    # hu.check_duplicates(exp_list_new)
    exp_list_new = hu.ignore_duplicates(exp_list_new)

    if return_style_list:
        return exp_list_new, style_list

    return exp_list_new
