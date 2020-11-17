
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

    try:
        db.latex_rows_widget = widgets.SelectMultiple(options=db.rm.exp_params,
                            value=list(db.vars.get('latex_rows')))
    except:
        db.latex_rows_widget = widgets.SelectMultiple(options=db.rm.exp_params,
                            value=[db.rm.exp_params[0]])

    metrics_txt = widgets.Label(value="Select Columns:", 
                                    layout=db.layout_label,)
    try:
        db.latex_cols_widget =  widgets.SelectMultiple(value=list(db.vars.get('latex_columns')),
                        options=[k for k in db.rm_original.score_keys if k is not 'None'])
    except:
        db.latex_cols_widget =  widgets.SelectMultiple(value=[db.rm_original.score_keys[0]],
                        options=[k for k in db.rm_original.score_keys if k is not 'None'])

    button = widgets.VBox([ 
                            widgets.HBox([hparam_txt, metrics_txt]),
                            widgets.HBox([db.latex_rows_widget, db.latex_cols_widget]),
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

            db.vars['latex_rows'] = list(db.latex_rows_widget.value)
            db.vars['latex_columns'] = list(db.latex_cols_widget.value)
            # print('cols', db.hparam_dict)
            # stop
            score_table = db.rm.get_latex_table(columns=db.vars.get('latex_columns'), 
                                            rows=db.vars.get('latex_rows'),
                                            caption='Results')
            print(score_table) 


    b_table.on_click(on_clicked)

    
    