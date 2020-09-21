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
# from . import tools


class ResultManager:
    def __init__(self, 
                 savedir_base,
                 exp_list=None,
                 filterby_list=None,
                 verbose=True,
                 has_score_list=False,
                 exp_groups=None,
                 mode_key=None,
                 exp_ids=None,
                 save_history=False,
                 score_list_name='score_list.pkl',
                 account_id=None):
        """[summary]
        
        Parameters
        ----------
        savedir_base : [type]
            A directory where experiments are saved
        exp_list : [type], optional
            [description], by default None
        filterby_list : [type], optional
            [description], by default None
        has_score_list : [type], optional
            [description], by default False
        
        Example
        -------
        >>> from haven import haven_results as hr
        >>> savedir_base='../results'
        >>> rm = hr.ResultManager(savedir_base=savedir_base,
                                filterby_list=[{'dataset':'mnist'}],
                                verbose=1)
        >>> for df in rm.get_score_df():
        >>>     display(df)
        >>> fig_list = rm.get_plot_all(y_metric_list=['train_loss', 'val_acc'], 
                                    order='groups_by_metrics',
                                    x_metric='epoch', 
                                    figsize=(15,6),
                                    title_list=['dataset'],
                                    legend_list=['model']) 
        """
        
        # sanity checks
        assert os.path.exists(savedir_base), '%s does not exist' % savedir_base

        self.exp_groups = {}
        if exp_groups is not None:
            if isinstance(exp_groups, dict):
                # load from a dict
                self.exp_groups = exp_groups
            elif os.path.exists(exp_groups):
                # load from a file
                self.exp_groups = hu.load_py(exp_groups).EXP_GROUPS
            else:
                raise ValueError('%s does not exist...' % exp_groups)
        
        # rest
        self.score_list_name = score_list_name
        self.mode_key = mode_key
        self.has_score_list = has_score_list
        self.save_history = save_history
        self.account_id = account_id
        # get exp _list
        
        if exp_ids is not None:
            assert exp_list is None, "settings exp_ids require exp_list=None"
            assert exp_groups is None, "settings exp_ids require exp_groups=None"
            exp_list = []
            for exp_id in exp_ids:
                exp_list += [hu.load_json(os.path.join(savedir_base, exp_id, 'exp_dict.json'))]
        else:
            if exp_list is None:
                exp_list = get_exp_list(savedir_base=savedir_base, verbose=verbose)
            else:
                exp_list = exp_list
            
            if len(exp_list) == 0:
                raise ValueError('exp_list is empty...')

        exp_list_with_scores = [e for e in exp_list if 
                                    os.path.exists(os.path.join(savedir_base, 
                                                                hu.hash_dict(e),
                                                                score_list_name))]
        if has_score_list:
            exp_list = exp_list_with_scores

        self.exp_list_all = copy.deepcopy(exp_list)

        
        self.score_keys  = ['None']

        if len(exp_list_with_scores):
            score_fname = os.path.join(savedir_base, hu.hash_dict(exp_list_with_scores[0]), score_list_name)
            self.score_keys = ['None'] + list(hu.load_pkl(score_fname)[0].keys())
                    
                                                        
        self.savedir_base = savedir_base
        
        self.filterby_list = filterby_list
        self.verbose = verbose

        self.n_exp_all = len(exp_list)

        self.exp_list = filter_exp_list(exp_list, 
                                        filterby_list=filterby_list, 
                                        savedir_base=savedir_base,
                                        verbose=verbose)

        if len(self.exp_list) != 0:
            self.exp_params = list(self.exp_list[0].keys())
        else:
            self.exp_params = []

        
        
        if mode_key:
            for exp_dict in exp_list:
                exp_dict[mode_key] = 1
            for exp_dict in self.exp_list_all:
                exp_dict[mode_key] = 1

        self.exp_groups['all'] = copy.deepcopy(self.exp_list_all)
    
    def get_state_dict(self):
        pass
    def load_state_dict(self, state_dict):
        pass

    def get_plot(self, groupby_list=None, savedir_plots=None, filterby_list=None, **kwargs):
        fig_list = []
        filterby_list = filterby_list or self.filterby_list
        exp_groups = group_exp_list(self.exp_list, groupby_list)
        
        for i, exp_list in enumerate(exp_groups):
            fig, ax = get_plot(exp_list=exp_list, savedir_base=self.savedir_base, filterby_list=filterby_list, 
                        
                        verbose=self.verbose,
                        score_list_name=self.score_list_name,
                               **kwargs)
            fig_list += [fig]

            # save image if given
            if savedir_plots != '' and savedir_plots is not None:
                os.makedirs(savedir_plots, exist_ok=True)
                save_fname = os.path.join(savedir_plots, "%d.png" % i )
                fig.savefig(save_fname, bbox_inches='tight')

        return fig_list

    def get_plot_all(self, y_metric_list, order='groups_by_metrics', 
                     groupby_list=None, ylim_list=None, xlim_list=None,
                     savedir_plots=None, legend_last_row_only=False, show_legend_all=None,
                     **kwargs):
        """[summary]
        
        Parameters
        ----------
        y_metric_list : [type]
            [description]
        order : str, optional
            [description], by default 'groups_by_metrics'
        
        Returns
        -------
        [type]
            [description]
        
        """
        
        if order not in ['groups_by_metrics', 'metrics_by_groups']:
            raise ValueError('%s order is not defined, choose between %s' % (order, ['groups_by_metrics', 'metrics_by_groups']))
        exp_groups = group_exp_list(self.exp_list, groupby_list)
        figsize = kwargs.get('figsize') or None
        
        fig_list = []

        if not isinstance(y_metric_list, list):
            y_metric_list = [y_metric_list]

        if ylim_list is not None:
            assert isinstance(ylim_list[0], list), "ylim_list has to be lists of lists"
        if xlim_list is not None:
            assert isinstance(xlim_list[0], list), "xlim_list has to be lists of lists"

        if order == 'groups_by_metrics':
            for j, exp_list in enumerate(exp_groups):   
                fig, ax_list = plt.subplots(nrows=1, ncols=len(y_metric_list), figsize=figsize)
                if not hasattr(ax_list, 'size'):
                    ax_list = [ax_list]
                for i, y_metric in enumerate(y_metric_list):
                    if i == (len(y_metric_list) - 1):
                        show_legend = True
                    else:
                        show_legend = False
                    
                    ylim = None
                    xlim = None
                    if ylim_list is not None:
                        assert len(ylim_list) == len(exp_groups), "ylim_list has to have %d rows" % len(exp_groups)
                        assert len(ylim_list[0]) == len(y_metric_list), "ylim_list has to have %d cols" % len(y_metric_list)
                        ylim = ylim_list[j][i]
                    if xlim_list is not None:
                        assert len(xlim_list) == len(exp_groups), "xlim_list has to have %d rows" % len(exp_groups)
                        assert len(xlim_list[0]) == len(y_metric_list), "xlim_list has to have %d cols" % len(y_metric_list)
                        xlim = xlim_list[j][i]
                    if show_legend_all is not None:
                        show_legend = show_legend_all
                    fig, _ = get_plot(exp_list=exp_list, savedir_base=self.savedir_base, y_metric=y_metric, 
                                    fig=fig, axis=ax_list[i], verbose=self.verbose, filterby_list=self.filterby_list,
                                    show_legend=show_legend,
                                    ylim=ylim, xlim=xlim,
                                    score_list_name=self.score_list_name,
                                    **kwargs)
                fig_list += [fig]

        elif order == 'metrics_by_groups':

            for j, y_metric in enumerate(y_metric_list):   
                fig, ax_list = plt.subplots(nrows=1, ncols=len(exp_groups) , figsize=figsize)
                if not hasattr(ax_list, 'size'):
                    ax_list = [ax_list]
                for i, exp_list in enumerate(exp_groups):
                    if i == 0:
                        show_ylabel = True
                    else:
                        show_ylabel = False

                    if i == (len(exp_groups) - 1):
                        show_legend = True
                    else:
                        show_legend = False

                    if legend_last_row_only and j < (len(y_metric_list) - 1):
                        show_legend = False

                    ylim = None
                    xlim = None
                    if ylim_list is not None:
                        assert len(ylim_list) == len(y_metric_list), "ylim_list has to have %d rows" % len(exp_groups)
                        assert len(ylim_list[0]) == len(exp_groups), "ylim_list has to have %d cols" % len(y_metric_list)
                        ylim = ylim_list[j][i]
                    if xlim_list is not None:
                        assert len(xlim_list) == len(y_metric_list), "xlim_list has to have %d rows" % len(exp_groups)
                        assert len(xlim_list[0]) == len(exp_groups), "xlim_list has to have %d cols" % len(y_metric_list)
                        xlim = xlim_list[j][i]

                    if show_legend_all is not None:
                        show_legend = show_legend_all
                        
                    fig, _ = get_plot(exp_list=exp_list, savedir_base=self.savedir_base, y_metric=y_metric, 
                                    fig=fig, axis=ax_list[i], verbose=self.verbose, filterby_list=self.filterby_list,
                                    ylim=ylim, xlim=xlim,
                                    show_legend=show_legend,
                                    show_ylabel=show_ylabel,
                                    score_list_name=self.score_list_name,
                                    **kwargs)
                fig_list += [fig]

        plt.tight_layout()
        if savedir_plots:
            for i in range(len(fig_list)):
                os.makedirs(savedir_plots, exist_ok=True)
                fname = os.path.join(savedir_plots + '_%d.pdf' % i)
                fig_list[i].savefig(fname,
                                     dpi=300, 
                                     bbox_inches='tight')
                print(fname, 'saved')
                
        return fig_list
    
    def get_score_df(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        df_list = get_score_df(exp_list=self.exp_list, 
                    savedir_base=self.savedir_base, verbose=self.verbose, 
                    score_list_name=self.score_list_name,
                    **kwargs)
        return df_list 

    def to_dropbox(self, outdir_base, access_token):
        """[summary]
        """ 
        hd.to_dropbox(self.exp_list, savedir_base=self.savedir_base, 
                          outdir_base=outdir_base,
                          access_token=access_token)

    def get_exp_list_df(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        df_list = get_exp_list_df(exp_list=self.exp_list,
                     verbose=self.verbose, **kwargs)
        
        return df_list 

    def get_exp_table(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        table = get_exp_list_df(exp_list=self.exp_list, verbose=self.verbose, **kwargs)
        return table 

     
    def get_score_table(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        table = get_score_df(exp_list=self.exp_list, 
                    savedir_base=self.savedir_base, 
                    score_list_name=self.score_list_name,
                    filterby_list=self.filterby_list,
                    verbose=self.verbose, **kwargs)
        return table 

    def get_score_lists(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        score_lists = get_score_lists(exp_list=self.exp_list, 
                                      savedir_base=self.savedir_base, 
                                      score_list_name=self.score_list_name,
                                      filterby_list=self.filterby_list,
                                      verbose=self.verbose, **kwargs)
        return score_lists

    def get_images(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return get_images(exp_list=self.exp_list, savedir_base=self.savedir_base, verbose=self.verbose, **kwargs)

    def get_job_summary(self, columns=None, add_prefix=False, **kwargs):
        """[summary]
        """
        exp_list = filter_exp_list(self.exp_list, self.filterby_list, savedir_base=self.savedir_base, verbose=self.verbose)
        jm = hjb.JobManager(exp_list=exp_list, savedir_base=self.savedir_base, account_id=self.account_id, **kwargs)
        summary_list = jm.get_summary_list(columns=columns, add_prefix=add_prefix)

        return summary_list
            
    def to_zip(self, savedir_base='', fname='tmp.zip', **kwargs):
        """[summary]
        
        Parameters
        ----------
        fname : [type]
            [description]
        """
        from haven.haven_share import haven_dropbox as hd
        if savedir_base == '':
            savedir_base = self.savedir_base
        exp_id_list = [hu.hash_dict(exp_dict) for exp_dict in self.exp_list]
        hd.zipdir(exp_id_list, savedir_base, fname, **kwargs)

    def to_dropbox(self, fname, dropbox_path=None, access_token=None):
        """[summary]
        
        Parameters
        ----------
        fname : [type]
            [description]
        dropbox_path : [type], optional
            [description], by default None
        access_token : [type], optional
            [description], by default None
        """
        from haven import haven_dropbox as hd

        out_fname = os.path.join(dropbox_path, fname)
        src_fname = os.path.join(self.savedir_base, fname)
        self.to_zip(src_fname)
        hd.upload_file_to_dropbox(src_fname, out_fname, access_token)
        print('saved: https://www.dropbox.com/home/%s' % out_fname)


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
    def split_func(x):
        x_list = []
        for k_list in groupby_list:
            if not isinstance(k_list, list):
                k_list = [k_list]
            val = get_str(x, k_list)
            x_list += [val]

        return x_list

    exp_list.sort(key=split_func)

    list_of_exp_list = []
    group_dict = groupby(exp_list, key=split_func)

    # exp_group_dict = {}
    for k, v in group_dict:
        v_list = list(v)
        list_of_exp_list += [v_list]
    #     # print(k)
    #     exp_group_dict['_'.join(list(map(str, k)))] = v_list

    return list_of_exp_list

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

def get_best_exp_dict(exp_list, savedir_base, metric, 
                      metric_agg='min', filterby_list=None, 
                      avg_across=None,
                      return_scores=False, verbose=True,
                        score_list_name='score_list.pkl'):
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
    if metric_agg in ['min', 'min_last']:
        best_score = np.inf
    elif metric_agg in ['max', 'max_last']:
        best_score = 0.
    
    exp_dict_best = None
    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)

        score_list_fname = os.path.join(savedir, score_list_name)
        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: missing %s' % (exp_id, score_list_name))
            continue
        
        score_list = hu.load_pkl(score_list_fname)

        if metric_agg in ['min', 'min_last']:
            if metric_agg == 'min_last':
                score = [score_dict[metric] for score_dict in score_list][-1]
            elif metric_agg == 'min':
                score = np.min([score_dict[metric] for score_dict in score_list])
            if best_score >= score:
                best_score = score
                exp_dict_best = exp_dict

        elif metric_agg in ['max', 'max_last']:
            if metric_agg == 'max_last':
                score = [score_dict[metric] for score_dict in score_list][-1]
            elif metric_agg == 'max':
                score = np.max([score_dict[metric] for score_dict in score_list])
                
            if best_score <= score:
                best_score = score
                exp_dict_best = exp_dict

        scores_dict += [{'score': score,
                         'epochs': len(score_list), 
                         'exp_id': exp_id}]

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
    assert(workdir is not None)
    if workdir not in sys.path:
        sys.path.append(workdir)
    import exp_configs as ec
    reload(ec)

    exp_list = []
    for exp_group in exp_group_list:
        exp_list += ec.EXP_GROUPS[exp_group]
    if verbose:
        print('%d experiments' % len(exp_list))

    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)
    return exp_list

def get_exp_list(savedir_base, filterby_list=None, verbose=True):
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
        fname = os.path.join(savedir, 'exp_dict.json')
        if len(exp_id) != 32:
            if verbose:
                print('"%s/" is not an exp directory' % exp_id)
            continue 

        if not os.path.exists(fname):
            if verbose:
                print('%s: missing exp_dict.json' % exp_id)
            continue
        
        exp_dict = hu.load_json(fname)
        expected_id = hu.hash_dict(exp_dict)
        if expected_id != exp_id:
            if verbose:
                # assert(hu.hash_dict(exp_dict) == exp_id)
                print('%s does not match %s' % (expected_id, exp_id))
            continue
        # print(hu.hash_dict(exp_dict))
        exp_list += [exp_dict]

    exp_list = filter_exp_list(exp_list, filterby_list)
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

def filter_exp_list(exp_list, filterby_list, savedir_base=None, verbose=True,
                    score_list_name='score_list.pkl', return_style_list=False):
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
    if filterby_list is None or filterby_list == '' or len(filterby_list) == 0:
        if return_style_list:
            return exp_list, [{}]*len(exp_list)
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
            
            if meta_dict.get('best'):
                assert savedir_base is not None
                el = filter_exp_list(exp_list, filterby_list=fd, verbose=verbose)
                best_dict = meta_dict.get('best')
                exp_dict = get_best_exp_dict(el, savedir_base, 
                                metric=best_dict['metric'],
                                metric_agg=best_dict['metric_agg'], 
                                filterby_list=None, 
                                avg_across=best_dict.get('avg_across'),
                                return_scores=False, 
                                verbose=verbose,
                                score_list_name=score_list_name)

                exp_list_new += [exp_dict]
                style_list += [meta_dict.get('style', {})]
            else:
                filterby_list_no_best += [filterby_dict] 

        
        # ignore metas here meta
        for exp_dict in exp_list:
            select_flag = False
            
            for fd in filterby_list_no_best:
                if isinstance(fd, tuple):
                    filterby_dict, meta_dict = fd
                    style_dict = meta_dict.get('style', {})
                else:
                    filterby_dict = fd
                    style_dict = {}

                filterby_dict = copy.deepcopy(filterby_dict)
               
                keys = filterby_dict.keys()
                for k in keys:
                    if '.' in k:
                        v = filterby_dict[k]
                        k_list = k.split('.')
                        nk = len(k_list)

                        dict_tree = dict()
                        t = dict_tree

                        for i in range(nk):
                            ki = k_list[i]
                            if i == (nk - 1):
                                t = t.setdefault(ki, v)
                            else:
                                t = t.setdefault(ki, {})

                        filterby_dict = dict_tree

                assert (isinstance(filterby_dict, dict), 
                                 ('filterby_dict: %s is not a dict' % str(filterby_dict)))

                if hu.is_subset(filterby_dict, exp_dict):
                    select_flag = True
                    break

            if select_flag:
                exp_list_new += [exp_dict]
                style_list += [style_dict]

        exp_list = exp_list_new
        

    if verbose:
        print('Filtered: %d/%d experiments gathered...' % (len(exp_list_new), len(exp_list)))
    # hu.check_duplicates(exp_list_new)
    exp_list_new = hu.ignore_duplicates(exp_list_new)
    
    if return_style_list:
        return exp_list_new, style_list

    return exp_list_new

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

    exp_list = filter_exp_list(exp_list, filterby_list, savedir_base=savedir_base, verbose=verbose)
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

    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)

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

def get_score_df(exp_list, savedir_base, filterby_list=None, columns=None,
                 score_columns=None,
                 verbose=True, wrap_size=8, hparam_diff=0, flatten_columns=True,
                 show_meta=True, show_max_min=True, add_prefix=False,
                 score_list_name='score_list.pkl', in_latex_format=False):
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
    exp_list = filter_exp_list(exp_list, filterby_list, savedir_base=savedir_base, verbose=verbose)

    # aggregate results
    result_list = []
    for exp_dict in exp_list:
        result_dict = {'creation_time':-1}

        exp_id = hu.hash_dict(exp_dict)
        if show_meta:
            # result_dict["exp_id"] = '\n'.join(wrap(exp_id, wrap_size))
            result_dict["exp_id"] = exp_id
        savedir = os.path.join(savedir_base, exp_id)
        score_list_fname = os.path.join(savedir, score_list_name)
        exp_dict_fname = os.path.join(savedir, "exp_dict.json")

        for k in exp_dict:
            if isinstance(columns, list) and k not in columns:
                continue
            if add_prefix:
                k_new = "(hparam) " + k
            else:
                k_new = k
            result_dict[k_new] = exp_dict[k]

        if os.path.exists(score_list_fname) and show_meta:
            result_dict['started_at'] = hu.time_to_montreal(exp_dict_fname)
            result_dict['creation_time'] = os.path.getctime(exp_dict_fname)

        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: %s is missing' % (exp_id, score_list_name))
            
        else:
            score_list = hu.load_pkl(score_list_fname)
            score_df = pd.DataFrame(score_list)
            if len(score_list):
                for k in score_df.columns:
                    if isinstance(score_columns, list) and k not in score_columns:
                        continue
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
        if flatten_columns:
            result_dict = hu.flatten_column(result_dict)

        result_list += [result_dict]

    # create table
    df = pd.DataFrame(result_list)

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

def get_result_dict(exp_dict, 
                    savedir_base, 
                    x_metric, 
                    y_metric,
                    exp_list=None, 
                    avg_across=False,
                    verbose=False,
                    plot_confidence=True,
                    x_cumsum=False,
                    score_list_name='score_list.pkl',
                    result_step=0):
    visited_exp_ids = set()
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    score_list_fname = os.path.join(savedir, score_list_name)

    # get scores
    if not avg_across:
        # get score list
        score_list = hu.load_pkl(score_list_fname)
        x_list = []
        y_list = []
        for score_dict in score_list:
            if x_metric in score_dict and y_metric in score_dict:
                x_list += [score_dict[x_metric]]
                y_list += [score_dict[y_metric]]
        y_std_list = []

    else:
        assert exp_list is not None, 'exp_list must be passed'
        # average score list across an hparam
        
        filter_dict = {k:exp_dict[k] for k in exp_dict if k not in avg_across}
        exp_sublist = filter_exp_list(exp_list, 
                                        filterby_list=[filter_dict], 
                                        savedir_base=savedir_base,
                                        verbose=verbose)
        def count(d):
            return sum([count(v) if isinstance(v, dict) 
                            else 1 for v in d.values()])
        n_values = count(filter_dict) + 1
        exp_sublist = [sub_dict for sub_dict in exp_sublist
                            if n_values == count(sub_dict)]
        # get score list
        x_dict = {}
        
        uniques= np.unique([sub_dict[avg_across] for sub_dict in exp_sublist])
        # print(uniques, len(exp_sublist))
        assert(len(exp_sublist)>0)
        assert(len(uniques) == len(exp_sublist))
        for sub_dict in exp_sublist:
            sub_id = hu.hash_dict(sub_dict)
            sub_score_list_fname = os.path.join(savedir_base, sub_id, score_list_name)

            if not os.path.exists(sub_score_list_fname):
                if verbose:
                    print('%s: %s does not exist...' % (sub_id, score_list_name))
                continue

            visited_exp_ids.add(sub_id)

            sub_score_list = hu.load_pkl(sub_score_list_fname)

            for score_dict in sub_score_list:
                if x_metric in score_dict and y_metric in score_dict:
                    x_val = score_dict[x_metric]
                    if not x_val in x_dict:
                        x_dict[x_val] = []

                    x_dict[x_val] += [score_dict[y_metric]]
        # import ipdb; ipdb.set_trace()
        if len(x_dict) == 0:
            x_list = []
            y_list = []
        else:
            x_list = np.array(list(x_dict.keys()))
            y_list_raw = list(x_dict.values())
            y_list_raw = [yy for yy in y_list_raw if len(yy) == len(exp_sublist)]
            y_list_all = np.array(y_list_raw)
            x_list = x_list[:len(y_list_all)]
            if y_list_all.dtype == 'object' or len(y_list_all)==0:
                x_list = []
                y_list = []
                y_std_list = []
            else:
                if plot_confidence:
                    y_std_list = np.std(y_list_all, axis=1)
                else:
                    y_std_list = 0
                y_list = np.mean(y_list_all, axis=1)  
                
    if x_cumsum:
        x_list = np.cumsum(x_list)
    
    if result_step == 0:
        return {'y_list':y_list, 
                'x_list':x_list,
                'y_std_list':y_std_list,
                'visited_exp_ids':visited_exp_ids}
    else:
        return {'y_list':y_list[::result_step], 
                'x_list':x_list[::result_step],
                'y_std_list':y_std_list,
                'visited_exp_ids':visited_exp_ids}


def get_plot(exp_list, savedir_base, 
             x_metric, y_metric,
             mode='line',
             filterby_list=None,
             title_list=None,
             legend_list=None,
             log_metric_list=None,
             figsize=None,
             avg_across=None,
             fig=None,
             axis=None,
             ylim=None,
             xlim=None,
             legend_fontsize=None,
             y_fontsize=None,
             x_fontsize=None,
             ytick_fontsize=None,
             xtick_fontsize=None,
             title_fontsize=None,
             legend_kwargs=None,
             map_title_list=tuple(),
             map_xlabel_list=tuple(),
             map_ylabel_list=dict(),
             bar_agg='min',
             verbose=True,
             show_legend=True,
             legend_format=None,
             title_format=None,
             cmap=None,
             show_ylabel=True,
             plot_confidence=True,
             x_cumsum=False,
             score_list_name='score_list.pkl',
             result_step=0,
             map_legend_list=dict()):
    """Plots the experiment list in a single figure.
    
    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : str
        A directory where experiments are saved
    x_metric : str
        Specifies metric for the x-axis
    y_metric : str
        Specifies metric for the y-axis
    title_list : [type], optional
        [description], by default None
    legend_list : [type], optional
        [description], by default None
    meta_list : [type], optional
        [description], by default None
    log_metric_list : [type], optional
        [description], by default None
    figsize : tuple, optional
        [description], by default (8, 8)
    avg_metric : [type], optional
        [description], by default None
    axis : [type], optional
        [description], by default None
    ylim : [type], optional
        [description], by default None
    xlim : [type], optional
        [description], by default None
    legend_fontsize : [type], optional
        [description], by default None
    y_fontsize : [type], optional
        [description], by default None
    ytick_fontsize : [type], optional
        [description], by default None
    xtick_fontsize : [type], optional
        [description], by default None
    legend_kwargs : [type], optional
        [description], by default None
    title_format: [str], optional
        [description], formatting of the title, by default None
    cmap: [str], optional
        [description], specify colormap, by default None
    
    Returns
    -------
    fig : [type]
        [description]
    axis : [type]
        [description]

    Examples
    --------
    >>> from haven import haven_results as hr
    >>> savedir_base='../results/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base, 
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> hr.get_plot(exp_list, savedir_base=savedir_base, x_metric='epoch', y_metric='train_loss', legend_list=['model'])
    """
    exp_list, style_list = filter_exp_list(exp_list, filterby_list=filterby_list,
                             savedir_base=savedir_base, 
                             verbose=verbose,
                             return_style_list=True)
    # if len(exp_list) == 0:
    if axis is None:
        fig, axis = plt.subplots(nrows=1, ncols=1,
                                    figsize=figsize)
    # default properties
    if title_list is not None:
        title = get_label(title_list, exp_list[0], format_str=title_format)
    else:
        title = ''

    
    ylabel = y_metric
    xlabel = x_metric

    # map properties
    for map_dict in map_title_list:
        if title in map_dict:
            title = map_dict[title]

    for map_dict in map_xlabel_list:
        if x_metric in map_dict:
            xlabel = map_dict[x_metric]

    for map_dict in map_ylabel_list:
        if y_metric in map_dict:
            ylabel = map_dict[y_metric]

    # set properties
    axis.set_title(title, fontsize=title_fontsize)
    if ylim is not None:
        axis.set_ylim(ylim)
    if xlim is not None:
        axis.set_xlim(xlim)
    
    if log_metric_list and y_metric in log_metric_list:
        axis.set_yscale('log')
        ylabel = ylabel + ' (log)'
    
    if log_metric_list and x_metric in log_metric_list:
        axis.set_xscale('log')
        xlabel = xlabel + ' (log)'

    if show_ylabel:
        axis.set_ylabel(ylabel, fontsize=y_fontsize)

    axis.set_xlabel(xlabel, fontsize=x_fontsize)

    axis.tick_params(axis='x', labelsize=xtick_fontsize)
    axis.tick_params(axis='y', labelsize=ytick_fontsize)

    axis.grid(True)

    # if len(exp_list) > 50:
    #     if verbose:
    #         raise ValueError('many experiments in one plot...use filterby_list to reduce them')
    # if cmap is not None or cmap is not '':
    #     plt.rcParams["axes.prop_cycle"] = get_cycle(cmap)
    if mode == 'pretty_plot':
        tools.pretty_plot
    
    bar_count = 0
    visited_exp_ids = set()
    for exp_dict, style_dict in zip(exp_list, style_list):
        exp_id = hu.hash_dict(exp_dict)
        if exp_id in visited_exp_ids:
            continue

        savedir = os.path.join(savedir_base, exp_id)
        score_list_fname = os.path.join(savedir, score_list_name)

        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: %s does not exist...' % (exp_id,score_list_name))
            continue

        else:
            result_dict = get_result_dict(exp_dict, 
                            savedir_base, 
                            x_metric, 
                            y_metric,
                            plot_confidence=plot_confidence,
                            exp_list=exp_list, 
                            avg_across=avg_across,
                            verbose=verbose,
                            x_cumsum=x_cumsum,
                            score_list_name=score_list_name,
                            result_step=result_step)
            
            y_list = result_dict['y_list']
            x_list = result_dict['x_list']
            for eid in list(result_dict['visited_exp_ids']):
                visited_exp_ids.add(eid)
            if len(x_list) == 0 or np.array(y_list).dtype == 'object':
                x_list = np.NaN
                y_list = np.NaN
                if verbose:
                    print('%s: "(%s, %s)" not in score_list' % (exp_id, y_metric, x_metric))

            # map properties of exp
            if legend_list is not None:
                label = get_label(legend_list, exp_dict, format_str=legend_format)
            else:
                label = exp_id

            color = None
            marker = '*'
            linewidth = None
            markevery = None
            markersize = None

            if len(style_dict):
                marker = style_dict.get('marker', marker)
                label = style_dict.get('label', label)
                color = style_dict.get('color', color)
                linewidth = style_dict.get('linewidth', linewidth)
                markevery = style_dict.get('markevery', markevery)
                markersize = style_dict.get('markersize', markersize)

            if label in map_legend_list:
                label = map_legend_list[label]
            # plot
            if mode == 'pretty_plot':
                # plot the mean in a line
                # pplot = pp.add_yxList
                axis.plot(x_list, y_list, color=color, linewidth=linewidth, markersize=markersize,
                    label=str(label), marker=marker, markevery=markevery)
                # tools.pretty_plot
            elif mode == 'line':
                # plot the mean in a line
                axis.plot(x_list, y_list, color=color, linewidth=linewidth, markersize=markersize,
                    label=label, marker=marker, markevery=markevery)

                if avg_across and hasattr(y_list, 'size'):
                    # add confidence interval
                    axis.fill_between(x_list, 
                            y_list - result_dict.get('y_std_list', 0),
                            y_list + result_dict.get('y_std_list', 0), 
                            color = color,  
                            alpha=0.1)

            elif mode == 'bar':
                # plot the mean in a line
                if bar_agg == 'max':
                    y_agg = np.max(y_list)    
                elif bar_agg == 'min':
                    y_agg = np.min(y_list)
                elif bar_agg == 'mean':
                    y_agg = np.mean(y_list)
                elif bar_agg == 'last':
                    y_agg = [y for y in y_list if isinstance(y, float)][-1]
                    
                    
                width = 0.
                import math
                
                if math.isnan(y_agg):
                    s = 'NaN'
                    continue
                    
                else:
                    s="%.3f" % y_agg

                axis.bar([bar_count + width], 
                        [y_agg],
                        color=color,
                        label=label,
                        # label='%s - (%s: %d, %s: %.3f)' % (label, x_metric, x_list[-1], y_metric, y_agg)
                        )
                if color is not None:
                    bar_color = color
                else:
                    bar_color = 'black'

                # minimum, maximum = axis.get_ylim()
                # y_height = .05 * (maximum - minimum)

                # axis.text(bar_count, y_agg + .01, "%.3f"%y_agg, color=bar_color, fontweight='bold')
                axis.text(x=bar_count, y = y_agg*1.01, 
                            s=s, 
                            fontdict=dict(fontsize=(y_fontsize or 12)), color='black', 
                            fontweight='bold')
                axis.set_xticks([])
                bar_count += 1
            else:
                raise ValueError('mode %s does not exist. Options: (line, bar)' % mode)

    legend_kwargs = legend_kwargs or {"loc":2, "bbox_to_anchor":(1.05,1),
                                      'borderaxespad':0., "ncol":1}
    
    if mode == 'pretty_plot':
        pass
    elif show_legend:
        axis.legend(fontsize=legend_fontsize, **legend_kwargs)

    plt.tight_layout()
    

    return fig, axis

def get_label(original_list, exp_dict, format_str=None):
    label_list = []
    for k in original_list:
        depth_list = k.split('.')
        sub_dict = exp_dict
        for d in depth_list:
            if sub_dict is None or d not in sub_dict:
                sub_dict = None
                break
            sub_dict = sub_dict[d]
            
        label_list += [str(sub_dict)]
        
    if format_str:
        label = format_str.format(*label_list)
    else:
        label = '_'.join(label_list)
    
    label = '\n'.join(wrap(label, 40))
    return label

def get_images(exp_list, savedir_base, n_exps=20, n_images=1,
                   figsize=(12,12), legend_list=None,
                   dirname='images', verbose=True):
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
                print('displayed %d/%d experiment images' % (k, n_exps))
            break

        result_dict = {}
        if legend_list is None:
            label = hu.hash_dict(exp_dict)
        else:
            label = '_'.join([str(exp_dict.get(k)) for
                                k in legend_list])

        exp_id = hu.hash_dict(exp_dict)
        result_dict['exp_id'] = exp_id
        if verbose:
            print('Displaying Images for Exp:', exp_id)
        savedir = os.path.join(savedir_base, exp_id)

        base_dir = os.path.join(savedir, dirname)
        img_list = glob.glob(os.path.join(base_dir, '*.jpg'))
        img_list += glob.glob(os.path.join(base_dir, '*.png'))

        img_list.sort(key=os.path.getmtime)
        img_list = img_list[::-1]
        img_list = img_list[:n_images]

        if len(img_list) == 0:
            if verbose:
                print('no images in %s' % base_dir)
            continue

        ncols = len(img_list)
        # ncols = len(exp_configs)
        nrows = 1

        print('%s\nExperiment id: %s' % ("="*100, exp_id,))
        for i in range(ncols):
            img_fname = os.path.split(img_list[i])[-1]
            fig = plt.figure(figsize=figsize)
            try:
                img = plt.imread(img_list[i])
                plt.imshow(img)
                plt.title('%s\n%s:%s' %
                                (exp_id, label, img_fname))

                plt.axis('off')
                plt.tight_layout()
                fig_list += [fig]

            except:
                print('skipping - %s, image corrupted' % img_fname)
            
            
        exp_count += 1
    
    return fig_list


