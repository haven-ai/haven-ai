import matplotlib
import numpy as np
import os
import pandas as pd
import pylab as plt

from itertools import product
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from sklearn.metrics.pairwise import pairwise_distances


matplotlib.style.use("bmh")

markers = [
    ("-", "o"),
    ("-", "p"),
    ("-", "D"),
    ("-", "^"),
    ("-", "s"),
    ("-", "8"),
    ("-", "o"),
    ("-", "o"),
    ("-", "o"),
    ("-", "o"),
    ("-", "o"),
    ("-", "o"),
]

colors = [
    "#741111",
    "#000000",
    "#3a49ba",
    "#7634c9",
    "#4C9950",
    "#CC29A3",
    "#ba3a3a",
    "#0f7265",
    "#7A7841",
    "#00C5CD",
    "#6e26d9",
]

bright_colors = ["#00C5CD"]


def myticks(x, pos):

    if x == 0:
        return "$0$"

    exponent = int(np.log10(x))
    coeff = x / 10 ** exponent

    # return r"{:0.1f}e{:0d}".format(coeff,exponent)

    return r"${:0.1f} \times 10^{{ {:2d} }}$".format(coeff, exponent)


def myticks_new(x, pos, exponent=1e5):

    if x == 0:
        return "$0$"

    exponent = int(np.log10(x))
    coeff = x / 10 ** exponent

    return r"${:0s}$".format(coeff / exponent)

    # return r"${:0.1f} \times 10^{{ {:2d} }}$".format(coeff,exponent)


class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of
    magnitude"""

    def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)

    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag


class PrettyPlot:
    def __init__(
        self,
        title=None,
        ylabel=None,
        xlabel=None,
        fontsize=14,
        linewidth=2.5,
        markersize=12,
        axFontSize=18,
        figsize=(13, 10),
        legend_type="line",
        yscale="log",
        subplots=(1, 1),
        shareRowLabel=True,
        axTickSize=14,
        legend_size=10,
        box_linewidth=1,
    ):
        self.box_linewidth = box_linewidth
        self.legend_size = legend_size
        self.axTickSize = int(axTickSize)
        self.fontsize = int(fontsize)
        self.shareRowLabel = shareRowLabel
        self.lim_set = False
        self.ylim = None
        self.legend_type = legend_type
        self.yscale = yscale
        self.linewidth = int(linewidth)
        self.markersize = int(markersize)
        self.axFontSize = int(axFontSize)

        if self.yscale == "log":
            plt.yscale("log")
        # ax.set_yscale('logit')
        self.labels = []
        self.y_list = []
        self.x_list = []
        self.converged = []
        fig = plt.figure(figsize=figsize)

        if title is not None:
            fig.suptitle(title, fontsize=self.axFontSize)
        self.fig = fig

        subplots = list(subplots)
        self.nrows = subplots[0]
        self.ncols = subplots[1]
        self.pIndex = 1

        self.axList = []

    def add_yxList(self, y_vals, x_vals, label, converged=False):
        if isinstance(y_vals, list):
            y_vals = np.array(y_vals)
        if isinstance(x_vals, list):
            x_vals = np.array(x_vals)

        self.y_list += [y_vals]
        self.x_list += [x_vals]

        self.labels += [label]

        self.converged += [converged]

    def show(self):
        plt.show()

    def save(self, path, iformat="png"):
        create_dirs(path)
        fname = path + ".%s" % iformat
        self.fig.savefig(fname, bbox_inches="tight")
        print(("Figure saved in %s" % (fname)))

    def plot_DataFrame(self, results):
        n_points, n_labels = results.shape

        x_vals = np.arange(n_points)
        labels = results.columns
        y_array = np.array(results)
        y_list = []
        x_list = []
        for j in range(n_labels):
            x_list += [x_vals]
            y_list += [y_array[:, j]]

        self.plot(y_list, x_list, labels)

    def set_lim(self, ylim, xlim):
        self.lim_set = True
        self.ylim = ylim

        self.ax.set_ylim(ylim)
        self.ax.set_xlim(xlim)

    def set_tickSize(self, labelsize=8):
        [tick.label.set_fontsize(labelsize) for tick in self.ax.yaxis.get_major_ticks()]
        [tick.label.set_fontsize(labelsize) for tick in self.ax.xaxis.get_major_ticks()]

    def set_title(self, title):
        self.fig.suptitle(title, fontsize=self.axFontSize, y=1.08)

    def plot(self, y_list=None, x_list=None, labels=None, ax=None, ylabel="", xlabel="", yscale=False):
        fig = self.fig

        if y_list == None and x_list == None:
            y_list = self.y_list
            x_list = self.x_list

        if yscale == "log":
            # Makse sure everything is non-negative
            # for yi in y_list:
            #     assert np.all(yi >= 0)

            # Set zeros to eps
            for i in range(len(y_list)):
                y_list[i] = np.maximum(y_list[i], np.finfo(float).eps)

            # Set zeros to eps
            for i in range(len(y_list)):

                opt_ind = np.where(y_list[i] == np.finfo(float).eps)[0]
                if opt_ind.size > 0:
                    opt_ind = opt_ind[0]

                    y_list[i] = y_list[i][: opt_ind + 1]
                    x_list[i] = x_list[i][: opt_ind + 1]

        n_labels = len(y_list)

        if ax is None:
            ax = self.fig.add_subplot(self.nrows, self.ncols, self.pIndex)

        ax.set_facecolor("white")
        ax.set_yscale("log", nonposy="clip")
        if labels is None and self.labels is None:
            labels = list(map(str, np.arange(n_labels)))
        elif labels is None:
            labels = self.labels

        ref_points = []
        for i in range(len(self.converged)):
            if self.converged[i] is not None:

                ref_points += [[self.converged[i]["X"], self.converged[i]["Y"]]]

        label_positions, label_indices = get_labelPositions(
            y_list, x_list, self.ylim, labels=labels, ref_points=np.array(ref_points)
        )

        ls_markers = markers

        if not self.lim_set:
            y_min, y_max = get_min_max(y_list)
            x_min, x_max = get_min_max(x_list)
            # y_min = max(y_min, 1e-8)
            ax.set_ylim([y_min, y_max])
            ax.set_xlim([x_min, x_max])

        for i in range(n_labels):
            color = colors[i]
            ls, marker = ls_markers[i]

            y_vals = y_list[i]
            x_vals = x_list[i]

            n_points = len(y_vals)

            label = labels[i]

            markerFreq = n_points / (int(np.log(n_points)) + 1)

            # SCATTER PLOT OPTIMAL
            # ind_opt = np.where(y_vals == np.finfo(float).eps)[0]

            # if ind_opt.size > 0:
            #     x_opt = x_vals[np.where(y_vals == np.finfo(float).eps)[0][0]]
            #     y_opt = np.finfo(float).eps

            if self.converged[i] is not None:
                ax.scatter(
                    self.converged[i]["X"],
                    self.converged[i]["Y"],
                    s=300,
                    marker="*",
                    color=color,
                    clip_on=False,
                    zorder=100,
                )
            ##
            (line,) = ax.plot(
                x_vals,
                y_vals,
                markevery=int(markerFreq),
                markersize=int(self.markersize),
                color=color,
                lw=self.linewidth,
                alpha=1.0,
                label=label,
                ls=ls,
                marker=marker,
            )

            if self.legend_type == "line":
                x_point, y_point = label_positions[i]
                angle = get_label_angle(x_vals, y_vals, label_indices[i], ax, color="0.5", size=12)

                box = dict(
                    facecolor="white",
                    edgecolor=color,
                    linestyle=ls,
                    # hatch=marker,
                    linewidth=int(2),
                    boxstyle="round",
                )

                ax.text(
                    x_point,
                    y_point,
                    label,
                    va="center",
                    ha="center",
                    rotation=angle,
                    color=color,
                    bbox=box,
                    fontsize=self.legend_size,
                )

            else:
                plt.legend(loc="best")

        if self.shareRowLabel and (((self.pIndex - 1) % (self.ncols)) == 0):
            ax.set_ylabel(ylabel, fontsize=self.axFontSize)

        if not self.shareRowLabel:
            ax.set_ylabel(ylabel, fontsize=self.axFontSize * 1.1)

        ax.set_xlabel(xlabel, fontsize=self.axFontSize * 1.1)

        ax.tick_params(labelsize=self.axTickSize * 1.3)
        ax.tick_params(axis="y", labelsize=int(self.axTickSize * 1.5))
        self.y_list = []
        self.x_list = []
        self.labels = []
        self.converged = []

        self.pIndex += 1
        self.axList += [ax]

        ax.minorticks_off()
        vals = np.logspace(np.log10(y_min), np.log10(y_max), 5)
        ax.set_yticks(vals)

        ax.yaxis.set_major_formatter(FuncFormatter(myticks))

        return fig, ax


#########
# HELPERS
#########
def get_overlapPercentage(index, y_list):
    n_points = y_list[0].size
    for i in range(index + 1):
        n_points = min(n_points, y_list[i].size)

    y_vector = y_list[index][:n_points, np.newaxis]

    prev_lines = np.zeros((n_points, index))

    for i in range(index):
        prev_lines[:, i] = y_list[i][:n_points]

        prev_lines /= np.linalg.norm(prev_lines, axis=0) + 1e-10

    y_norm = y_vector / np.linalg.norm(y_vector, axis=0)

    diff = np.abs((prev_lines - y_norm)).min(axis=1)

    n_overlap = np.sum(diff < 1e-6)

    percentage = n_overlap / float(n_points)

    return percentage


def create_dirs(fname):
    if "/" not in fname:
        return

    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass


def normalize(xy_points, ref_points, y_min, y_max, x_min, x_max):

    xy_points[:, 1] = np.log(np.maximum(1e-15, xy_points[:, 1])) / np.log(10)
    ref_points[:, 1] = np.log(np.maximum(ref_points[:, 1], 1e-15)) / np.log(10)

    y_min = np.log(y_min) / np.log(10)
    y_max = np.log(y_max) / np.log(10)

    xy_normed = xy_points - np.array([x_min, y_min])
    xy_normed /= np.array([x_max - x_min, y_max - y_min])

    ref_normed = ref_points - np.array([x_min, y_min])

    ref_normed /= np.array([x_max - x_min, y_max - y_min])
    return xy_normed, ref_normed


# LABEL POSITIONS


def get_labelPositions(y_list, x_list, ylim=None, labels=None, ref_points=None):
    if ref_points is None:

        ref_points = []
    """Get label positions greedily"""

    n_labels = len(y_list)

    # GET BORDER POINTS
    x_min, x_max = get_min_max(x_list)
    if ylim is not None:
        y_min, y_max = ylim
    else:
        y_min, y_max = get_min_max(y_list)

    xdiff = x_max - x_min
    ydiff = y_max - y_min

    # Border points
    bp1 = np.array(list(product([x_min, x_max, xdiff * 0.5], [y_min, y_max, ydiff * 0.5])))[:-1]
    bp1 = np.array(list(product([x_max], [y_max])))[:-1]

    bp1 = np.array(list(product([8], [0])))

    addedPoints = []

    for yPoint in np.linspace(y_min, y_max, 6):
        addedPoints += [(x_min, yPoint)]
        addedPoints += [(x_max, yPoint)]

    sPoints = [(xx[0], yy[0]) for xx, yy in zip(x_list, y_list)]
    ePoints = [(xx[-1], yy[-1]) for xx, yy in zip(x_list, y_list)]
    bp2 = np.array(addedPoints + sPoints + ePoints)
    if len(ref_points) == 0:
        border_points = np.vstack([bp1, bp2])
    else:
        border_points = np.vstack([bp1, bp2, ref_points])

    n_border = border_points.shape[0]

    # Initialize placeholders
    ref_points = np.zeros((n_border + n_labels, 2))

    label_positions = np.zeros((n_labels, 2))
    label_indices = np.zeros(n_labels, int)

    ref_points[:n_border] = border_points

    for i in range(n_labels):
        # GET POSITIONS

        if ylim is not None:
            ind = (y_list[i] < y_max + 1e-4) & (y_list[i] > y_min - 1e-4)
            n_points = x_list[i][ind].size
            xy_points = np.zeros((n_points, 2))
            xy_points[:, 0] = x_list[i][ind]
            xy_points[:, 1] = y_list[i][ind]
        else:
            n_points = x_list[i].size
            xy_points = np.zeros((n_points, 2))
            xy_points[:, 0] = x_list[i]
            xy_points[:, 1] = y_list[i]

        # NORMALIZE

        xy_normed, ref_normed = normalize(
            xy_points.copy(), ref_points[: n_border + i].copy(), y_min, y_max, x_min, x_max
        )
        # GET REF POINTS

        dist = pairwise_distances(xy_normed, ref_normed, metric="l1")

        # GET MINIMUM DISTANCES
        min_dist = dist.min(axis=1)

        # GET MAXIMUM MINIMUM DISTANCE
        label_index = np.argmax(min_dist)
        label_pos = xy_points[label_index]

        ref_points[n_border + i] = label_pos
        label_positions[i] = label_pos
        label_indices[i] = label_index

    return label_positions, label_indices


def get_min_max(v_list):
    vector = v_list[0]
    v_min = np.min(vector)
    v_max = np.max(vector)

    for i in range(1, len(v_list)):
        vector = v_list[i]
        v_min = min(np.min(vector), v_min)
        v_max = max(np.max(vector), v_max)

    return v_min, v_max


def get_label_angle(xdata, ydata, index, ax, color="0.5", size=12, window=3):
    n_points = xdata.size

    x1 = xdata[index]
    y1 = ydata[index]

    # ax = line.get_axes()

    sp1 = ax.transData.transform_point((x1, y1))

    slope_degrees = 0.0
    count = 0.0

    for i in range(index + 1, min(index + window, n_points)):
        y2 = ydata[i]
        x2 = xdata[i]

        sp2 = ax.transData.transform_point((x2, y2))

        rise = sp2[1] - sp1[1]
        run = sp2[0] - sp1[0]

        slope_degrees += np.degrees(np.arctan2(rise, run))
        count += 1.0

    for i in range(index - 1, max(index - window, 0), -1):
        y2 = ydata[i]
        x2 = xdata[i]

        sp2 = ax.transData.transform_point((x2, y2))

        rise = -(sp2[1] - sp1[1])
        run = -(sp2[0] - sp1[0])

        slope_degrees += np.degrees(np.arctan2(rise, run))
        count += 1.0

    slope_degrees /= count

    return slope_degrees


def box_color(edgecolor, linestyle, marker):
    """Creates box shape"""
    return dict(
        facecolor="white",
        edgecolor=edgecolor,
        linestyle=linestyle,
        # hatch=marker,
        linewidth=2,
        boxstyle="round",
    )
