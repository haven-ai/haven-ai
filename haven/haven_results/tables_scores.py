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

from .. import haven_jobs as hjb
from .. import haven_utils as hu
from .. import haven_share as hd

def get_score_df(exp_list, savedir_base, filterby_list=None, columns=None,
                 score_columns=None,
                 verbose=True, wrap_size=8, hparam_diff=0, flatten_columns=True,
                 show_meta=True, show_max_min=True, add_prefix=False,
                 score_list_name='score_list.pkl', in_latex_format=False, avg_across=None):
    """Get a table showing the scores for the given list of experiments 

    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    columns : list, optional
        a list of columns you would like to display, by default None
    savedir_base : str, optional
        A directory where experiments are saved
        
    Returns
    -------
    DataFrame
        a dataframe showing the scores obtained by the experiments

    Example
    -------
    >>> from haven import haven_results as hr
    >>> savedir_base='../results/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base, 
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> df = hr.get_score_df(exp_list, savedir_base=savedir_base, columns=['train_loss', 'exp_id'])
    >>> print(df)
    """
    if len(exp_list) == 0:
        if verbose:
            print('exp_list is empty...')
        return
    exp_list = hu.filter_exp_list(exp_list, filterby_list, savedir_base=savedir_base, verbose=verbose)

    # aggregate results
    result_list = []
    for exp_dict in exp_list:
        result_dict = {}
        exp_id = hu.hash_dict(exp_dict)
        if avg_across is not None:
            tmp_dict = copy.deepcopy(exp_dict)
            del tmp_dict[avg_across]
            result_dict['_' + avg_across] = hu.hash_dict(tmp_dict)

        savedir = os.path.join(savedir_base, exp_id)
        score_list_fname = os.path.join(savedir, score_list_name)
        exp_dict_fname = os.path.join(savedir, "exp_dict.json")

        hparam_columns = columns or list(exp_dict.keys())
        for k in hparam_columns:
            if add_prefix:
                k_new = "(hparam) " + k
            else:
                k_new = k
            result_dict[k_new] = exp_dict[k]

        if os.path.exists(score_list_fname) and show_meta:
            result_dict['started_at'] = hu.time_to_montreal(exp_dict_fname)
            result_dict['creation_time'] = os.path.getctime(exp_dict_fname)
        else:
            result_dict['creation_time'] = -1

        if show_meta:
            result_dict["exp_id"] = exp_id

        if flatten_columns:
            result_dict = hu.flatten_column(result_dict)

        hparam_columns = list(result_dict.keys())

        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: %s is missing' % (exp_id, score_list_name))
            
        else:
            score_list = hu.load_pkl(score_list_fname)
            score_df = pd.DataFrame(score_list)
            metric_columns = score_columns or score_df.columns
            if len(score_list):
                for k in metric_columns:
                    v = np.array(score_df[k])
                    if 'float' in str(v.dtype):
                        v = v[~np.isnan(v)]

                    if len(v):
                        if add_prefix:
                            k_new = "(metric) " +k
                        else:
                            k_new = k

                        if "float" in str(v.dtype):
                            result_dict[k_new] = v[-1]
                            if show_max_min:
                                result_dict[k_new+' (max)'] = v.max()
                                result_dict[k_new+' (min)'] = v.min()
                                
                        else:
                            result_dict[k_new] = v[-1]
        

        result_list += [result_dict]

    # create table
    df = pd.DataFrame(result_list)
    metric_columns = [c for c in result_dict if c not in hparam_columns]
    # print(avg_across)
    if avg_across is not None:
        df_avg = df.groupby('_' + avg_across)
        df_tmp = df_avg[hparam_columns].first().join(df_avg[metric_columns].agg([np.mean, np.std]))
        del df_tmp['_' + avg_across]
        df_tmp = df_tmp.reset_index()
        del df_tmp['_' + avg_across]
        # del df_tmp[avg_across]
        df = df_tmp 

    df = df.sort_values(by='creation_time' )
    del df['creation_time']

    # wrap text for prettiness
    df = hu.pretty_print_df(df)

    if hparam_diff > 0 and len(df) > 1:
        cols = hu.get_diff_columns(df, min_threshold=hparam_diff, max_threshold='auto')
        df = df[cols]

    df =  hu.sort_df_columns(df)
    if in_latex_format:
        return df.to_latex(index=False)
    return df

def get_score_lists(exp_list, savedir_base, filterby_list=None, verbose=True,
                    score_list_name='score_list.pkl'):
    """[summary]
    
    Parameters
    ----------
    exp_list : [type]
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : [type]
        [description]
    filterby_list : [type], optional
        [description], by default None
    
    Returns
    -------
    [type]
        [description]
    Example
    -------
    >>> from haven import haven_results as hr
    >>> savedir_base='../results/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base, 
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> lists_of_score_lists = hr.get_score_lists(exp_list, savedir_base=savedir_base, columns=['train_loss', 'exp_id'])
    >>> print(lists_of_score_lists)
    """
    if len(exp_list) == 0:
        if verbose:
            print('exp_list is empty...')
        return

    exp_list = hu.filter_exp_list(exp_list, filterby_list, savedir_base=savedir_base, verbose=verbose)
    score_lists = []

    # aggregate results
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)

        score_list_fname = os.path.join(savedir, score_list_name)
        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: missing %s' % (exp_id, score_list_name))
            continue
        
        else:
            score_lists += [hu.load_pkl(score_list_fname)]
    
    return score_lists


def get_exp_list_df(exp_list, filterby_list=None, columns=None, verbose=True):
    """Get a table showing the configurations for the given list of experiments 

    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    columns : list, optional
        a list of columns you would like to display, by default None
        
    Returns
    -------
    DataFrame
        a dataframe showing the scores obtained by the experiments

    Example
    -------
    >>> from haven import haven_results as hr
    >>> savedir_base='../results/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base, 
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> df = hr.get_exp_list_df(exp_list, columns=['train_loss', 'exp_id'])
    >>> print(df)
    """
    if len(exp_list) == 0:
        if verbose:
            print('exp_list is empty...')
        return

    exp_list = hu.filter_exp_list(exp_list, filterby_list, verbose=verbose)

    # aggregate results
    result_list = []
    for exp_dict in exp_list:
        result_dict = {}

        exp_id = hu.hash_dict(exp_dict)
        result_dict["exp_id"] = exp_id

        for k in exp_dict:
            result_dict[k] = exp_dict[k]

        result_list += [result_dict]

    df = pd.DataFrame(result_list)

    if columns:
        df = df[[c for c in columns if c in df.columns]]

    return df