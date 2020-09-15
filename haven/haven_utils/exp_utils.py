import os 
import copy
import hashlib
import itertools


def cartesian_exp_group(exp_config):
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
    exp_list_raw = (dict(zip(exp_config_copy.keys(), values))
                    for values in itertools.product(*exp_config_copy.values()))

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        exp_list += [exp_dict]

    return exp_list


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
        raise ValueError('exp_dict is not a dict')

    for k in sorted(exp_dict.keys()):
        if '.' in k:
            raise ValueError(". has special purpose")
        elif isinstance(exp_dict[k], dict):
            v = hash_dict(exp_dict[k])
        elif isinstance(exp_dict[k], tuple):
            raise ValueError("tuples can't be hashed yet, consider converting tuples to lists")
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