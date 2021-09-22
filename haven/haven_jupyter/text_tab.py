from .. import haven_utils
from .. import haven_results as hr
from .. import haven_utils as hu
from .. import haven_share as hd

import os, glob
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


def text_tab(db, output):
    blogs = widgets.Button(description="Get Predictions")

    button = widgets.VBox(
        [
            widgets.HBox([blogs]),
        ]
    )
    output_plot = widgets.Output()

    with output:
        display(button)
        display(output_plot)

    def on_logs_clicked(b):
        output_plot.clear_output()
        with output_plot:
            text_list = []
            for exp_dict in db.rm.exp_list:
                exp_id = hu.hash_dict(exp_dict)
                savedir_text = os.path.join(db.rm.savedir_base, exp_id, "text")
                if os.path.exists(savedir_text):
                    path = os.path.join(savedir_text, "*.json")
                    text_list = [hu.load_json(glob.glob(path)[0])]
                else:
                    text_list += ["None"]

            n_logs = len(text_list)

            for i, (exp_dict, text) in enumerate(zip(db.rm.exp_list, text_list)):
                exp_id = hu.hash_dict(exp_dict)
                print("\nPredictions %d/%d" % (i + 1, n_logs), "=" * 50)
                print("exp_id:", exp_id)
                print("savedir:", os.path.join(db.rm_original.savedir_base, exp_id))

                print("\nexp_dict")
                print("-" * 50)
                pprint.pprint(exp_dict)

                print("\nPredictions")
                print("-" * 50)
                pprint.pprint(text)

    blogs.on_click(on_logs_clicked)
