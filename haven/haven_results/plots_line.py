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


def get_plot(
    exp_list,
    savedir_base,
    x_metric,
    y_metric,
    mode="line",
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
    legend_fontsize=12,
    y_fontsize=14,
    x_fontsize=14,
    ytick_fontsize=12,
    xtick_fontsize=12,
    title_fontsize=18,
    legend_kwargs=None,
    map_title_list=tuple(),
    map_xlabel_list=tuple(),
    map_ylabel_list=dict(),
    bar_agg="min",
    verbose=True,
    show_legend=True,
    legend_format=None,
    title_format=None,
    cmap=None,
    show_ylabel=True,
    plot_confidence=True,
    x_cumsum=False,
    score_list_name="score_list.pkl",
    result_step=0,
    map_legend_list=dict(),
    xticks=None,
):
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
    exp_list, style_list = hu.filter_exp_list(
        exp_list, filterby_list=filterby_list, savedir_base=savedir_base, verbose=verbose, return_style_list=True
    )
    if legend_list is None:
        legend_list = hu.get_diff_hparam(exp_list)
    # if len(exp_list) == 0:
    if axis is None:
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    # default properties
    if title_list is not None:
        title = get_label(title_list, exp_list[0], format_str=title_format)
    else:
        title = ""

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
        axis.set_yscale("log")
        ylabel = ylabel + " (log)"

    if log_metric_list and x_metric in log_metric_list:
        axis.set_xscale("log")
        xlabel = xlabel + " (log)"

    if show_ylabel:
        axis.set_ylabel(ylabel, fontsize=y_fontsize)

    axis.set_xlabel(xlabel, fontsize=x_fontsize)

    axis.tick_params(axis="x", labelsize=xtick_fontsize)
    axis.tick_params(axis="y", labelsize=ytick_fontsize)

    axis.grid(True)

    # if len(exp_list) > 50:
    #     if verbose:
    #         raise ValueError('many experiments in one plot...use filterby_list to reduce them')
    # if cmap is not None or cmap is not '':
    #     plt.rcParams["axes.prop_cycle"] = get_cycle(cmap)
    if mode == "pretty_plot":
        tools.pretty_plot

    bar_count = 0
    visited_exp_ids = set()
    plot_idx = 0
    for exp_dict, style_dict in zip(exp_list, style_list):
        exp_id = hu.hash_dict(exp_dict)
        if exp_id in visited_exp_ids:
            continue

        savedir = os.path.join(savedir_base, exp_id)
        score_list_fname = os.path.join(savedir, score_list_name)

        # skipt if it does not exist
        if not os.path.exists(score_list_fname):
            if verbose:
                print("%s: %s does not exist..." % (exp_id, score_list_name))
            continue

        # get result
        result_dict = get_result_dict(
            exp_dict,
            savedir_base,
            x_metric,
            y_metric,
            plot_confidence=plot_confidence,
            exp_list=exp_list,
            avg_across=avg_across,
            verbose=verbose,
            x_cumsum=x_cumsum,
            score_list_name=score_list_name,
            result_step=result_step,
        )
        # it is None if one of score_list.pkl is corrupted
        if result_dict is None:
            continue

        y_list = result_dict["y_list"]
        x_list = result_dict["x_list"]
        for eid in list(result_dict["visited_exp_ids"]):
            visited_exp_ids.add(eid)
        if len(x_list) == 0 or np.array(y_list).dtype == "object":
            x_list = np.NaN
            y_list = np.NaN
            if verbose:
                print('%s: "(%s, %s)" not in score_list' % (exp_id, y_metric, x_metric))

        # map properties of exp
        if legend_list is not None:
            if plot_idx != 0:
                show_legend_key = False
            else:
                show_legend_key = True
            label = get_label(legend_list, exp_dict, format_str=legend_format, show_key=show_legend_key)
        else:
            label = exp_id

        plot_idx += 1

        color = None
        marker = "o"
        linewidth = 2.8

        try:
            markevery = len(x_list) // 10
        except Exception:
            markevery = None
        if markevery == 0:
            markevery = None

        markersize = 6

        if len(style_dict):
            marker = style_dict.get("marker", marker)
            label = style_dict.get("label", label)
            color = style_dict.get("color", color)
            linewidth = style_dict.get("linewidth", linewidth)
            markevery = style_dict.get("markevery", markevery)
            markersize = style_dict.get("markersize", markersize)

        if label in map_legend_list:
            label = map_legend_list[label]
        # plot
        if mode == "pretty_plot":
            # plot the mean in a line
            # pplot = pp.add_yxList
            axis.plot(
                x_list,
                y_list,
                color=color,
                linewidth=linewidth,
                markersize=markersize,
                label=str(label),
                marker=marker,
                markevery=markevery,
            )
            # tools.pretty_plot
        elif mode == "line":
            # plot the mean in a line
            (line_plot,) = axis.plot(
                x_list,
                y_list,
                color=color,
                linewidth=linewidth,
                markersize=markersize,
                label=label,
                marker=marker,
                markevery=markevery,
            )

            if avg_across and hasattr(y_list, "size"):
                # add confidence interval
                axis.fill_between(
                    x_list,
                    y_list - result_dict.get("y_std_list", 0),
                    y_list + result_dict.get("y_std_list", 0),
                    color=line_plot.get_color(),
                    alpha=0.1,
                )

        elif mode == "bar":
            # plot the mean in a line
            if bar_agg == "max":
                y_agg = np.max(y_list)
            elif bar_agg == "min":
                y_agg = np.min(y_list)
            elif bar_agg == "mean":
                y_agg = np.mean(y_list)
            elif bar_agg == "last":
                y_agg = [y for y in y_list if isinstance(y, float)][-1]

            width = 0.0
            import math

            if math.isnan(y_agg):
                s = "NaN"
                continue

            else:
                s = "%.3f" % y_agg

            axis.bar(
                [bar_count + width],
                [y_agg],
                color=color,
                label=label,
                # label='%s - (%s: %d, %s: %.3f)' % (label, x_metric, x_list[-1], y_metric, y_agg)
            )
            if color is not None:
                bar_color = color
            else:
                bar_color = "black"

            # minimum, maximum = axis.get_ylim()
            # y_height = .05 * (maximum - minimum)

            # axis.text(bar_count, y_agg + .01, "%.3f"%y_agg, color=bar_color, fontweight='bold')
            axis.text(
                x=bar_count,
                y=y_agg * 1.01,
                s=s,
                fontdict=dict(fontsize=(y_fontsize or 12)),
                color="black",
                fontweight="bold",
            )
            axis.set_xticks([])
            bar_count += 1
        else:
            raise ValueError("mode %s does not exist. Options: (line, bar)" % mode)

    legend_kwargs = legend_kwargs or {"loc": 2, "bbox_to_anchor": (1.05, 1), "borderaxespad": 0.0, "ncol": 1}

    if mode == "pretty_plot":
        pass
    elif show_legend:
        axis.legend(fontsize=legend_fontsize, **legend_kwargs)

    plt.tight_layout()
    if xticks is not None:
        plt.xticks(*xticks)

    return fig, axis


def get_label(original_list, exp_dict, format_str=None, show_key=False):
    key_list = []
    label_list = []
    for i, k in enumerate(original_list):
        if k == "exp_id":
            sub_dict = str(hu.hash_dict(exp_dict))
        else:
            depth_list = k.split(".")
            sub_dict = exp_dict
            for d in depth_list:
                if sub_dict is None or d not in sub_dict:
                    sub_dict = None
                    break
                sub_dict = sub_dict[d]

        if i < (len(original_list) - 1):
            if format_str:
                key_list += [f"{k}"]
                label_list += [f"{str(sub_dict):{int(len(k))}}"]
            else:
                key_list += [f"{k}|"]
                label_list += [f"{str(sub_dict):{int(len(k))}}|"]
        else:
            key_list += [f"{k}"]
            label_list += [f"{str(sub_dict)}"]

    if format_str:
        label = format_str.format(*label_list)
    else:
        label = " ".join(label_list)
        if show_key:
            label = " ".join(key_list) + "\n" + label

    # label = '\n'.join(wrap(label, 40))
    return label


def get_result_dict(
    exp_dict,
    savedir_base,
    x_metric,
    y_metric,
    exp_list=None,
    avg_across=False,
    verbose=False,
    plot_confidence=True,
    x_cumsum=False,
    score_list_name="score_list.pkl",
    result_step=0,
):
    visited_exp_ids = set()
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    score_list_fname = os.path.join(savedir, score_list_name)

    # get scores
    if not avg_across:
        # get score list
        try:
            score_list = hu.load_pkl(score_list_fname)
        except Exception:
            return None

        x_list = []
        y_list = []
        for score_dict in score_list:
            if x_metric in score_dict and y_metric in score_dict:
                x_list += [score_dict[x_metric]]
                y_list += [score_dict[y_metric]]
        y_std_list = []

    else:
        assert exp_list is not None, "exp_list must be passed"
        # average score list across an hparam

        filter_dict = {k: exp_dict[k] for k in exp_dict if k not in avg_across}
        exp_sublist = hu.filter_exp_list(
            exp_list, filterby_list=[filter_dict], savedir_base=savedir_base, verbose=verbose
        )

        def count(d):
            return sum([count(v) if isinstance(v, dict) else 1 for v in d.values()])

        n_values = count(filter_dict) + 1
        exp_sublist = [sub_dict for sub_dict in exp_sublist if n_values == count(sub_dict)]
        # get score list
        x_dict = {}

        uniques = np.unique([sub_dict[avg_across] for sub_dict in exp_sublist])
        # print(uniques, len(exp_sublist))
        assert len(exp_sublist) > 0
        assert len(uniques) == len(exp_sublist)
        for sub_dict in exp_sublist:
            sub_id = hu.hash_dict(sub_dict)
            sub_score_list_fname = os.path.join(savedir_base, sub_id, score_list_name)

            if not os.path.exists(sub_score_list_fname):
                if verbose:
                    print("%s: %s does not exist..." % (sub_id, score_list_name))
                continue

            visited_exp_ids.add(sub_id)

            try:
                sub_score_list = hu.load_pkl(sub_score_list_fname)
            except Exception:
                if verbose:
                    print("%s: %s is corrupt..." % (sub_id, score_list_name))
                return None

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
            x_list = x_list[: len(y_list_all)]
            if y_list_all.dtype == "object" or len(y_list_all) == 0:
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
        return {"y_list": y_list, "x_list": x_list, "y_std_list": y_std_list, "visited_exp_ids": visited_exp_ids}
    else:
        return {
            "y_list": y_list[::result_step],
            "x_list": x_list[::result_step],
            "y_std_list": y_std_list,
            "visited_exp_ids": visited_exp_ids,
        }
