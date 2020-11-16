
from .. import haven_utils
from .. import haven_results as hr
from .. import haven_utils as hu 
from .. import haven_share as hd

import os
import pprint, json
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


def latex_tab(db, output):

    b_table = widgets.Button(description="Display Latex Table")

    # d_avg_across_columns = widgets.Text(
    #     value=str(db.vars.get('avg_across', 'None')),
    #     description='avg_across:',
    #     disabled=False
    # )

    hparam_txt = widgets.Label(value="Select Rows:", 
                                    layout=widgets.Layout(width='300px'),)
    db.hparam_widget = widgets.SelectMultiple(options=db.rm.exp_params)

    metrics_txt = widgets.Label(value="Select Columns:", 
                                    layout=db.layout_label,)
    db.metrics_widget =  widgets.SelectMultiple(options=[k for k in db.rm_original.score_keys if k is not 'None'])

    button = widgets.VBox([ 
                            widgets.HBox([hparam_txt, metrics_txt]),
                            widgets.HBox([db.hparam_widget, db.metrics_widget]),
                            widgets.HBox([b_table]),
    ])
    output_plot = widgets.Output()

    with output:
        display(button)
        display(output_plot)

    def on_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()

            db.vars['columns'] = list(db.hparam_widget.value)
            db.vars['score_columns'] = list(db.metrics_widget.value)
            # print('cols', db.hparam_dict)
            # stop
            score_table = db.rm.get_latex_table(columns=db.vars.get('score_columns'), 
                                            rows=db.vars.get('columns'),
                                            caption='Results')
            print(score_table) 


    b_table.on_click(on_clicked)

    
    