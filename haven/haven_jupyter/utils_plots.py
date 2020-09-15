
from .. import haven_utils
from .. import haven_results as hr
from .. import haven_utils as hu
from .. import haven_share as hd

import os
import pprint
import json
import copy
import pprint
import pandas as pd

try:
    import ast
    from ipywidgets import Button, HBox, VBox
    from ipywidgets import widgets

    from IPython.display import display
    from IPython.core.display import Javascript, display, HTML
    from IPython.display import FileLink, FileLinks
    from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
except:
    print('widgets not available...')


def plots_tab(self, output):
    llegend_list = widgets.Text(
        value=str(self.vars.get('legend_list', '[model]')),
        description='legend_list:',
        disabled=False
    )
    llegend_format = widgets.Text(
        value=str(self.vars.get('legend_format', '')),
        description='legend_format:',
        disabled=False
    )
    ltitle_format = widgets.Text(
        value=str(self.vars.get('title_format', '')),
        description='title_format:',
        disabled=False
    )

    lcmap = widgets.Text(
        value=str(self.vars.get('cmap', 'jet')),
        description='cmap:',
        layout=self.layout_dropdown,
        disabled=False
    )

    llog_metric_list = widgets.Text(
        value=str(self.vars.get('log_metric_list', '[train_loss]')),
        description='log_metric_list:',
        disabled=False
    )

    t_y_metric = widgets.Text(
        value=str(self.vars.get('y_metrics', 'train_loss')),
        description='y_metrics:',
        disabled=False
    )

    d_x_metric_columns = widgets.Text(
        value=str(self.vars.get('x_metric', 'epoch')),
        description='x_metric:',
        disabled=False
    )

    t_groupby_list = widgets.Text(
        value=str(self.vars.get('groupby_list')),
        description='groupby_list:',
        disabled=False
    )

    t_mode = widgets.Text(
        value=str(self.vars.get('mode', 'line')),
        description='mode:',
        disabled=False
    )

    t_bar_agg = widgets.Text(
        value=str(self.vars.get('bar_agg', 'mean')),
        description='bar_agg:',
        disabled=False
    )

    t_title_list = widgets.Text(
        value=str(self.vars.get('title_list', 'dataset')),
        description='title_list:',
        disabled=False
    )

    d_style = widgets.Dropdown(
        options=['False', 'True'],
        value='False',
        description='interactive:',
        layout=self.layout_dropdown,
        disabled=False,
    )
    d_avg_across_txt = widgets.Label(value="avg_across:",
                                     layout=widgets.Layout(width='75px'),)

    d_avg_across_columns = widgets.Text(
        value=str(self.vars.get('avg_across', 'None')),
        description='avg_across:',
        disabled=False
    )

    bdownload = widgets.Button(description="Download Plots",
                               layout=self.layout_button)
    bdownload_out = widgets.Output(layout=self.layout_button)

    def on_download_clicked(b):
        fname = 'plots'
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        pp = PdfPages(fname)
        for fig in self.rm_original.fig_list:
            fig.savefig(pp, format='pdf')
        pp.close()

        bdownload_out.clear_output()

        with bdownload_out:
            display(FileLink(fname, result_html_prefix="Download: "))

    bdownload.on_click(on_download_clicked)

    brefresh = widgets.Button(description="Display Plot")
    button = widgets.VBox([widgets.HBox([brefresh, bdownload, bdownload_out]),
                           widgets.HBox([t_title_list, d_style]),
                           widgets.HBox([t_y_metric,  d_x_metric_columns]),
                           widgets.HBox([t_groupby_list, llegend_list, ]),
                           widgets.HBox([t_mode, t_bar_agg]),
                           widgets.HBox([ltitle_format, llegend_format]),
                           widgets.HBox([d_avg_across_columns]),

                           ])

    output_plot = widgets.Output()

    def on_clicked(b):
        if d_style.value == 'True':
            from IPython import get_ipython
            ipython = get_ipython()
            ipython.magic("matplotlib widget")
        output_plot.clear_output()
        with output_plot:
            self.update_rm()

            self.vars['y_metrics'] = hu.get_list_from_str(t_y_metric.value)
            self.vars['x_metric'] = d_x_metric_columns.value

            w, h = 10, 5
            if len(self.vars['y_metrics']) > 1:
                figsize = (2*int(w), int(h))
                self.vars['figsize'] = figsize
            else:
                self.vars['figsize'] = (int(w), int(h))

            self.vars['legend_list'] = hu.get_list_from_str(llegend_list.value)
            self.vars['legend_format'] = llegend_format.value
            self.vars['log_metric_list'] = hu.get_list_from_str(
                llog_metric_list.value)
            self.vars['groupby_list'] = hu.get_list_from_str(
                t_groupby_list.value)
            self.vars['mode'] = t_mode.value
            self.vars['title_list'] = hu.get_list_from_str(t_title_list.value)
            self.vars['bar_agg'] = t_bar_agg.value
            self.vars['title_format'] = ltitle_format.value
            self.vars['cmap'] = lcmap.value
            self.vars['avg_across'] = d_avg_across_columns.value

            avg_across_value = self.vars['avg_across']
            if avg_across_value == "None":
                avg_across_value = None

            self.rm_original.fig_list = self.rm.get_plot_all(y_metric_list=self.vars['y_metrics'],
                                                             x_metric=self.vars['x_metric'],
                                                             groupby_list=self.vars['groupby_list'],
                                                             legend_list=self.vars['legend_list'],
                                                             log_metric_list=self.vars['log_metric_list'],
                                                             mode=self.vars['mode'],
                                                             bar_agg=self.vars['bar_agg'],
                                                             figsize=self.vars['figsize'],
                                                             title_list=self.vars['title_list'],
                                                             legend_format=self.vars['legend_format'],
                                                             title_format=self.vars['title_format'],
                                                             cmap=self.vars['cmap'],
                                                             avg_across=avg_across_value)

            show_inline_matplotlib_plots()

    d_style.observe(on_clicked)
    brefresh.on_click(on_clicked)

    with output:
        display(button)
        display(output_plot)
