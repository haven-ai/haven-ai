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
from . import widgets as wdg

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


def images_tab(self, output):
    db = self

    if db.vars.get("legend_list") is None:
        db.vars["legend_list"] = hu.get_diff_hparam(db.rm.exp_list)
    w_legend = wdg.SelectMultiple(header="Legend:", options=db.rm.exp_params, db_vars=db.vars, var="legend_list")

    w_n_exps = wdg.Text("n_exps:", default="3", type="int", db_vars=db.vars, var="n_exps")
    w_n_images = wdg.Text("n_images:", default="5", type="int", db_vars=db.vars, var="n_images")
    w_figsize = wdg.Text("figsize:", default="(10,5)", type="tuple", db_vars=db.vars, var="figsize")
    w_dirname = wdg.Text("dirname:", default="images", type="str", db_vars=db.vars, var="dirname")

    bdownload = widgets.Button(description="Download Images", layout=self.layout_button)
    bdownload_out = widgets.Output(layout=self.layout_button)
    brefresh = widgets.Button(description="Display Images")
    button = widgets.VBox(
        [
            widgets.HBox(
                [
                    w_legend.get_widget(),
                    w_n_exps.get_widget(),
                    w_n_images.get_widget(),
                    w_figsize.get_widget(),
                    w_dirname.get_widget(),
                ]
            ),
            widgets.HBox([brefresh, bdownload, bdownload_out]),
        ]
    )

    output_plot = widgets.Output()

    with output:
        display(button)
        display(output_plot)

    def on_clicked(b):
        output_plot.clear_output()
        with output_plot:
            self.update_rm()

            self.rm_original.fig_image_list = self.rm.get_images(
                legend_list=w_legend.update(),
                n_images=w_n_images.update(),
                n_exps=w_n_exps.update(),
                figsize=w_figsize.update(),
                dirname=w_dirname.update(),
            )
            show_inline_matplotlib_plots()

    brefresh.on_click(on_clicked)

    def on_download_clicked(b):
        fname = "images"
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        pp = PdfPages(fname)
        for fig in self.rm_original.fig_image_list:
            fig.savefig(pp, format="pdf")
        pp.close()

        bdownload_out.clear_output()

        with bdownload_out:
            display(FileLink(fname, result_html_prefix="Download: "))

    bdownload.on_click(on_download_clicked)
