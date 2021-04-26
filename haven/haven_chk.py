import shutil
import os
import torch

from . import haven_utils as hu


def delete_experiment(savedir, backup_flag=False):
    """Delete an experiment. If the backup_flag is true it moves the experiment
    to the delete folder.

    Parameters
    ----------
    savedir : str
        Directory of the experiment
    backup_flag : bool, optional
        If true, instead of deleted is moved to delete folder, by default False
    """
    # get experiment id
    exp_id = os.path.split(savedir)[-1]
    assert len(exp_id) == 32

    # get paths
    savedir_base = os.path.dirname(savedir)
    savedir = os.path.join(savedir_base, exp_id)

    if backup_flag:
        # create 'deleted' folder
        dst = os.path.join(savedir_base, "deleted", exp_id)
        os.makedirs(dst, exist_ok=True)

        if os.path.exists(dst):
            shutil.rmtree(dst)

    if os.path.exists(savedir):
        if backup_flag:
            # moves folder to 'deleted'
            shutil.move(savedir, dst)
        else:
            # delete experiment folder
            shutil.rmtree(savedir)

    # make sure the experiment doesn't exist anymore
    assert not os.path.exists(savedir)


def load_checkpoint(exp_dict, savedir_base, fname="model_best.pth"):
    path = os.path.join(savedir_base, hu.hash_dict(exp_dict), fname)
    return torch.load(path)


def delete_and_backup_experiment(savedir):
    """Delete an experiment and make a backup (Movo to the trash)

    Parameters
    ----------
    savedir : str
        Directory of the experiment
    """
    # delete and backup experiment
    delete_experiment(savedir, backup_flag=True)


def get_savedir(exp_dict, savedir_base):
    """[summary]

    Parameters
    ----------
    exp_dict : dict
        Dictionary describing the hyperparameters of an experiment
    savedir_base : str
        Directory where the experiments are saved

    Returns
    -------
    str
        Directory of the experiment
    """
    # get experiment savedir
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    return savedir


def copy_checkpoints(savedir_base, filterby_list, hparam, hparam_new):
    """"""
    # get experiment savedir
    import copy
    import tqdm
    from . import haven_utils as hr

    exp_list = hr.get_exp_list(savedir_base=savedir_base)
    exp_list_new = hr.filter_exp_list(exp_list, filterby_list=filterby_list)
    exp_list_out = copy.deepcopy(exp_list_new)
    for exp_dict in exp_list_out:
        exp_dict[hparam_new] = exp_dict[hparam]
        del exp_dict[hparam]

    for e1, e2 in tqdm.tqdm(zip(exp_list_new, exp_list_out)):
        h1 = hu.hash_dict(e1)
        s1 = os.path.join(args.savedir_base, h1)

        h2 = hu.hash_dict(e2)
        s2 = os.path.join(args.savedir_base, h2)

        # copy exp dict
        os.makedirs(s2, exist_ok=True)
        e2_fname = os.path.join(s2, "exp_dict.json")
        hu.save_json(e2_fname, e2)

        for fname in ["score_list.pkl", "borgy_dict.json", "job_dict.json", "logs.txt", "err.txt"]:

            # copy score list
            s1_fname = os.path.join(s1, fname)
            s2_fname = os.path.join(s2, fname)
            # assert(not os.path.exists(s_list_fname))
            if os.path.exists(s1_fname):
                if ".json" in fname:
                    hu.save_json(s2_fname, hu.load_json(s1_fname))
                elif ".pkl" in fname:
                    hu.save_pkl(s2_fname, hu.load_pkl(s1_fname))
                elif ".txt" in fname:
                    hu.save_txt(s2_fname, hu.load_txt(s1_fname)[-5:])
