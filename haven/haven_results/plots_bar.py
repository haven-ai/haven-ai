import numpy as np
import pylab as plt


def get_bar_chart(score_list, label_list, sep, ylabel, fontsize, title, width, legend_flag=False, figsize=(20, 10)):
    """
    This function is used to plot the bar chart.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ind = np.arange(len(score_list)) * sep
    plt.title(title, fontsize=fontsize + 4)
    width = width
    for i in range(len(ind)):
        index = ind[i]
        value = score_list[i]

        rects = ax.bar([index + width], [value], width=width, label=label_list[i])

    minimum, maximum = ax.get_ylim()
    y = 0.05 * (maximum - minimum)

    for i in range(len(ind)):
        index = ind[i]
        value = score_list[i]

        ax.text(x=index + width - 0.025, y=y, s=f"{value}", fontdict=dict(fontsize=(fontsize or 10) + 2), color="white")

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    ax.set_ylabel(ylabel, fontsize=fontsize + 2)
    ax.grid(True)

    if legend_flag:
        plt.legend(**{"fontsize": fontsize, "loc": 2, "bbox_to_anchor": (1.05, 1), "borderaxespad": 0.0, "ncol": 1})
    else:
        ax.set_xticks(ind + width)
        ax.set_xticklabels(label_list)
    plt.tight_layout()

    plt.show()
