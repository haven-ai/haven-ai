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


def latex_tab(db, output):

    b_table = widgets.Button(description="Display Latex Table")

    w_rows = wdg.SelectMultiple(header="Select Rows:", options=db.rm.exp_params, db_vars=db.vars, var="latex_rows")
    w_cols = wdg.SelectMultiple(
        header="Select Columns:", options=db.rm.score_keys, db_vars=db.vars, var="latex_columns"
    )

    button = widgets.VBox(
        [
            widgets.HBox([w_rows.get_widget(), w_cols.get_widget()]),
            widgets.HBox([b_table]),
        ]
    )
    output_plot = widgets.Output()

    with output:
        display(button)
        display(output_plot)

    def on_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()

            score_table = db.rm.get_latex_table(columns=w_cols.update(), rows=w_rows.update(), caption="Results")
            print(score_table)

    b_table.on_click(on_clicked)
