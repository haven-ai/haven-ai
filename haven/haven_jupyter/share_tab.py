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


def share_tab(self, output):
    with output:
        ldownload = widgets.Label(
            value="Get 'score_list.pkl' and 'exp_dict' for each experiment.",
            #    layout=self.layout_button
        )
        bdownload = widgets.Button(description="Download Results", layout=self.layout_button)

        bdownload_out = widgets.Output(layout=self.layout_button)

        # bdownload_dropbox = widgets.Button(description="Upload to Dropbox",
        #                         layout=self.layout_button)

        # bdownload_out_dropbox  = widgets.Output(layout=self.layout_button)

        l_fname_list = widgets.Text(
            value=str(self.vars.get("fname_list", "")),
            layout=self.layout_dropdown,
            description="fname_list:",
            disabled=False,
        )

        # l_dropbox_path = widgets.Text(
        #     value=str(self.vars.get('dropbox_path', '/shared')),
        #     description='dropbox_path:',
        #     layout=self.layout_dropdown,
        #     disabled=False
        #         )
        # l_access_token_path = widgets.Text(
        #     value=str(self.vars.get('access_token', '')),
        #     description='access_token:',
        #     layout=self.layout_dropdown,
        #     disabled=False
        #         )
        def on_upload_clicked(b):
            fname = "results.zip"
            bdownload_out_dropbox.clear_output()
            self.vars["fname_list"] = hu.get_list_from_str(l_fname_list.value)
            self.vars["dropbox_path"] = l_dropbox_path.value
            self.vars["access_token"] = l_access_token_path.value
            with bdownload_out_dropbox:
                self.rm.to_zip(
                    savedir_base="",
                    fname=fname,
                    fname_list=self.vars["fname_list"],
                    dropbox_path=self.vars["dropbox_path"],
                    access_token=self.vars["access_token"],
                )

            os.remove("results.zip")
            display("result.zip sent to dropbox at %s." % self.vars["dropbox_path"])

        def on_download_clicked(b):
            fname = "results.zip"
            bdownload_out.clear_output()
            self.vars["fname_list"] = hu.get_list_from_str(l_fname_list.value)

            with bdownload_out:
                self.rm.to_zip(savedir_base="", fname=fname, fname_list=self.vars["fname_list"])
            bdownload_out.clear_output()
            with bdownload_out:
                display("%d exps zipped." % len(self.rm.exp_list))
            display(FileLink(fname, result_html_prefix="Download: "))

        bdownload.on_click(on_download_clicked)
        bdownload_zip = widgets.VBox([bdownload, bdownload_out])
        # bdownload_dropbox.on_click(on_upload_clicked)
        # bdownload_dropbox_vbox = widgets.VBox([ bdownload_dropbox, bdownload_out_dropbox])
        display(
            widgets.VBox(
                [
                    # widgets.HBox([l_fname_list, l_dropbox_path, l_access_token_path]),
                    ldownload,
                    widgets.HBox([bdownload_zip]),
                ]
            )
        )
