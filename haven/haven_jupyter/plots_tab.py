from .. import haven_utils
from .. import haven_results as hr
from .. import haven_utils as hu
from .. import haven_share as hd
from . import widgets as wdg

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
except Exception:
    print("widgets not available...")


def plots_tab(self, output):
    db = self

    llegend_format = widgets.Text(value=str(self.vars.get("legend_format", "")), description="", disabled=False)
    ltitle_format = widgets.Text(value=str(self.vars.get("title_format", "")), description="", disabled=False)

    lcmap = widgets.Text(
        value=str(self.vars.get("cmap", "jet")), description="cmap:", layout=self.layout_dropdown, disabled=False
    )

    llog_metric_list = widgets.Text(
        value=str(self.vars.get("log_metric_list", "[train_loss]")), description="log_metric_list:", disabled=False
    )

    bdownload = widgets.Button(description="Download Plots")
    bdownload_out = widgets.Output(layout=self.layout_button)

    def on_download_clicked(b):
        fname = "plots.pdf"
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        pp = PdfPages(fname)
        for fig in self.rm_original.fig_list:
            fig.savefig(pp, format="pdf")
        pp.close()

        bdownload_out.clear_output()

        with bdownload_out:
            display(FileLink(fname, result_html_prefix="Download: "))

    bdownload.on_click(on_download_clicked)

    h22 = widgets.Label(
        value="Format:",
        layout=widgets.Layout(width="340px"),
    )

    h33 = widgets.Label(
        value="Format:",
        layout=widgets.Layout(width="340px"),
    )

    h44 = widgets.Label(
        value="",
        layout=widgets.Layout(width="340px"),
    )

    space = widgets.Label(
        value="",
        layout=widgets.Layout(width="300px"),
    )
    brefresh = widgets.Button(description="Display Plot")
    d_avg_across_txt = widgets.Label(
        value="avg_across:",
    )

    w_y_metrics = wdg.SelectMultiple(
        header="Y-axis Metrics:", options=self.rm_original.score_keys, db_vars=db.vars, var="y_metrics"
    )

    if db.vars.get("legend_list") is None:
        db.vars["legend_list"] = hu.get_diff_hparam(db.rm.exp_list)
    w_legend = wdg.SelectMultiple(
        header="Legend:", options=["exp_id"] + db.rm.exp_params, db_vars=db.vars, var="legend_list", select_all=True
    )

    w_title = wdg.SelectMultiple(header="Title:", options=db.rm.exp_params, db_vars=db.vars, var="title_list")

    w_groupby = wdg.SelectMultiple(
        header="GroupBy:", options=["None"] + db.rm.exp_params, db_vars=db.vars, var="groupby_list"
    )

    w_x_metric = wdg.Dropdown(
        header="X-axis Metric", options=self.rm_original.score_keys, db_vars=db.vars, var="x_metric"
    )
    w_mode = wdg.Dropdown(header="Plot Mode", options=["line", "bar"], db_vars=db.vars, var="mode")
    w_bar_agg = wdg.Dropdown(
        header="Plot Agg (bar plot only) ", options=["last", "max", "mean", "min"], db_vars=db.vars, var="bar_agg"
    )
    w_avg_across = wdg.Dropdown(
        header="Avg Across", options=["None"] + db.rm.exp_params, db_vars=db.vars, var="avg_across"
    )

    button = widgets.VBox(
        [
            widgets.HBox(
                [
                    w_y_metrics.get_widget(),
                    w_legend.get_widget(),
                    w_title.get_widget(),
                    w_groupby.get_widget(),
                ]
            ),
            widgets.HBox(
                [
                    w_x_metric.get_widget(),
                    w_mode.get_widget(),
                    w_bar_agg.get_widget(),
                    w_avg_across.get_widget(),
                ]
            ),
            #    widgets.HBox([ d_avg_across_txt, d_avg_across_columns,  ]),
            widgets.HBox(
                [
                    brefresh,
                    bdownload,
                    bdownload_out,
                ]
            ),
        ]
    )

    output_plot = widgets.Output()

    def on_clicked(b):
        output_plot.clear_output()
        with output_plot:
            self.update_rm()

            w, h = 10, 5
            if len(w_y_metrics.update()) == 0:
                display("No results saved yet.")
                return

            elif len(w_y_metrics.update()) > 1:
                figsize = (2 * int(w), int(h))
                self.vars["figsize"] = figsize
            else:
                self.vars["figsize"] = (int(w), int(h))

            self.vars["legend_format"] = llegend_format.value
            self.vars["log_metric_list"] = hu.get_list_from_str(llog_metric_list.value)

            self.vars["title_format"] = ltitle_format.value
            self.vars["cmap"] = lcmap.value

            self.rm_original.fig_list = self.rm.get_plot_all(
                y_metric_list=w_y_metrics.update(),
                x_metric=w_x_metric.update(),
                groupby_list=w_groupby.update(),
                legend_list=w_legend.update(),
                log_metric_list=self.vars["log_metric_list"],
                mode=w_mode.update(),
                bar_agg=w_bar_agg.update(),
                figsize=self.vars["figsize"],
                title_list=w_title.update(),
                legend_format=self.vars["legend_format"],
                title_format=self.vars["title_format"],
                cmap=self.vars["cmap"],
                avg_across=w_avg_across.update(),
            )

            show_inline_matplotlib_plots()

    brefresh.on_click(on_clicked)

    with output:
        display(button)
        display(output_plot)
