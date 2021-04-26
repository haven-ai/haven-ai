import copy
import glob
import os
import sys
import pprint
from itertools import groupby
from textwrap import wrap
import numpy as np
import pandas as pd
import pylab as plt
import tqdm
from . import plots_line
from .. import haven_utils as hu

try:
    from IPython.display import Image
    from IPython.display import display
except Exception:
    pass


def get_images(
    exp_list, savedir_base, n_exps=20, n_images=1, figsize=(12, 12), legend_list=None, dirname="images", verbose=True
):
    """[summary]

    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : str
        A directory where experiments are saved
    n_exps : int, optional
        [description], by default 3
    n_images : int, optional
        [description], by default 1
    height : int, optional
        [description], by default 12
    width : int, optional
        [description], by default 12
    legend_list : [type], optional
        [description], by default None
    dirname : str, optional
        [description], by default 'images'
    Returns
    -------
    fig_list : list
        a list of pylab figures
    Example
    -------
    >>> from haven import haven_results as hr
    >>> savedir_base='../results/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base,
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> hr.get_images(exp_list, savedir_base=savedir_base)
    """
    fig_list = []
    exp_count = 0
    for k, exp_dict in enumerate(exp_list):

        if exp_count >= n_exps:
            if verbose:
                print("displayed %d/%d experiment images" % (k, n_exps))
            break

        result_dict = {}

        exp_id = hu.hash_dict(exp_dict)
        result_dict["exp_id"] = exp_id
        if verbose:
            print("Displaying Images for Exp:", exp_id)
        savedir = os.path.join(savedir_base, exp_id)

        base_dir = os.path.join(savedir, dirname)
        img_list = glob.glob(os.path.join(base_dir, "*.jpg"))
        img_list += glob.glob(os.path.join(base_dir, "*.png"))
        img_list += glob.glob(os.path.join(base_dir, "*.gif"))
        img_list.sort(key=os.path.getmtime)
        img_list = img_list[::-1]
        img_list = img_list[:n_images]

        if len(img_list) == 0:
            if verbose:
                print("no images in %s" % base_dir)
            continue

        ncols = len(img_list)
        # ncols = len(exp_configs)

        # from IPython.display import display
        # display('%s' % ("="*50))
        result_dict = {"exp_id": exp_id}
        result_dict.update(copy.deepcopy(exp_dict))
        score_list_path = os.path.join(savedir, "score_list.pkl")
        if os.path.exists(score_list_path):
            score_list = hu.load_pkl(score_list_path)
            result_dict.update(score_list[-1])
        # display(pd.Series(result_dict))
        if legend_list is not None:
            label = plots_line.get_label(legend_list, exp_dict, show_key=True)
        else:
            label = exp_id

        if "epoch" in result_dict:
            label += "_epoch:%d" % result_dict["epoch"]
        # if legend_list is None:
        #     label = hu.hash_dict(exp_dict)
        # else:
        #     label = '-'.join(['%s:%s' % (k, str(result_dict.get(k))) for
        #                         k in legend_list])

        for i in range(ncols):
            img_fname = os.path.split(img_list[i])[-1]
            title = f"{exp_id} - {img_fname}\n{label}"
            fig = plt.figure(figsize=figsize)

            if ".gif" in img_list[i]:

                display(exp_id)
                display(exp_dict)
                display(title)
                display(Image(img_list[i]))
            else:
                img = plt.imread(img_list[i])
                plt.imshow(img)
                # display(exp_id)
                # display(exp_dict)
                plt.title(title, fontsize=20)

                plt.axis("off")
                plt.tight_layout()
            fig_list += [fig]

        exp_count += 1

    return fig_list
