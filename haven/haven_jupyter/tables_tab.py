
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


def tables_tab(db, output):
    d_columns_txt = widgets.Label(value="Select Hyperparam column", 
                                    layout=db.layout_label,)
    d_columns = widgets.Dropdown(
                options=['None'] + db.rm.exp_params,
                value='None',
                layout=db.layout_dropdown,
                disabled=False,
            )
    d_score_columns_txt = widgets.Label(value="Select Score column",
                                        layout=db.layout_label,)
    d_score_columns = widgets.Dropdown(
            options=db.rm_original.score_keys,
            value='None',
            layout=db.layout_dropdown,
            disabled=False,
        )

    bstatus = widgets.Button(description="Jobs Status")
    blogs = widgets.Button(description="Jobs Logs")
    bfailed = widgets.Button(description="Jobs Failed")

    b_table = widgets.Button(description="Display Table")
    b_meta = widgets.Button(description="Display Meta Table")
    b_diff = widgets.Button(description="Display Filtered Table")

    # d_avg_across_columns = widgets.Text(
    #     value=str(db.vars.get('avg_across', 'None')),
    #     description='avg_across:',
    #     disabled=False
    # )
    d_avg_across_txt = widgets.Label(value="avg_across:",)
    d_avg_across_columns =  widgets.Dropdown(
                options=['None'] + db.rm.exp_params,
                value='None',
                layout=db.layout_dropdown,
                disabled=False,
            )
    hparam_txt = widgets.Label(value="Select hyperparamters:", 
                                    layout=widgets.Layout(width='300px'),)
    metrics_txt = widgets.Label(value="Select metrics:", 
                                    layout=db.layout_label,)
    db.hparam_widget = widgets.SelectMultiple(options=db.rm.exp_params)
    db.metrics_widget =  widgets.SelectMultiple(options=[k for k in db.rm_original.score_keys if k is not 'None'])

    button = widgets.VBox([widgets.HBox([hparam_txt, metrics_txt]),
                            widgets.HBox([db.hparam_widget, db.metrics_widget]),
                            widgets.HBox([b_table, bstatus, blogs, bfailed, d_avg_across_txt, d_avg_across_columns]),
                            # widgets.HBox([d_columns_txt, d_score_columns_txt]),
                            # widgets.HBox([d_columns, d_score_columns ]),
    ])
    output_plot = widgets.Output()

    with output:
        display(button)
        display(output_plot)

    def on_table_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()

            db.vars['avg_across'] = d_avg_across_columns.value
            avg_across_value = db.vars.get('avg_across', 'None')
            if avg_across_value == "None":
                avg_across_value = None

            db.vars['columns'] = list(db.hparam_widget.value)
            db.vars['score_columns'] = list(db.metrics_widget.value)
            # print('cols', db.hparam_dict)
            # stop
            score_table = db.rm.get_score_table(columns=db.vars.get('columns'), 
                                            score_columns=db.vars.get('score_columns'),
                                            avg_across=avg_across_value)
            display(score_table) 

    def on_job_status_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()
            summary_list = db.rm.get_job_summary(verbose=db.rm.verbose,
                                               add_prefix=True)
            summary_dict = hu.group_list(summary_list, key='job_state', return_count=True)
            display(summary_dict)

            summary_dict = hu.group_list(summary_list, key='job_state', return_count=False)

            for state in summary_dict:
                n_jobs = len(summary_dict[state])
                if n_jobs:
                    display('Experiments %s: %d' %(state, n_jobs))
                    df = pd.DataFrame(summary_dict[state])
                    display(df.head())

    def on_logs_clicked(b):
        output_plot.clear_output()
        with output_plot:
            summary_list = db.rm.get_job_summary(verbose=db.rm.verbose,
                                               add_prefix=True)
            
            n_logs = len(summary_list)
        
            for i, logs in enumerate(summary_list):
                print('\nLogs %d/%d' % (i+1, n_logs), '='*50)
                print('exp_id:', logs['exp_id'])
                print('job_id:', logs['job_id'])
                print('job_state:', logs['job_state'])
                print('savedir:', os.path.join(db.rm_original.savedir_base, logs['exp_id']))

                print('\nexp_dict')
                print('-'*50)
                pprint.pprint(logs['exp_dict'])
                
                print('\nLogs')
                print('-'*50)
                pprint.pprint(logs['logs'])     
    
    def on_failed_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()
            summary_list = db.rm.get_job_summary(verbose=db.rm.verbose,
                                               add_prefix=True)
            summary_dict = hu.group_list(summary_list, key='job_state', return_count=False)
            if 'FAILED' not in summary_dict:
                display('NO FAILED JOBS')
                return
            n_failed = len(summary_dict['FAILED'])
        
            if n_failed == 0:
                display('no failed experiments')
            else:
                for i, failed in enumerate(summary_dict['FAILED']):
                    print('\nFailed %d/%d' % (i+1, n_failed), '='*50)
                    print('exp_id:', failed['exp_id'])
                    print('job_id:', failed['job_id'])
                    print('job_state:', 'FAILED')
                    print('savedir:', os.path.join(db.rm_original.savedir_base, failed['exp_id']))

                    print('\nexp_dict')
                    print('-'*50)
                    pprint.pprint(failed['exp_dict'])
                    
                    print('\nLogs')
                    print('-'*50)
                    pprint.pprint(failed['logs'])

    # Add call listeners
    b_table.on_click(on_table_clicked)
    bstatus.on_click(on_job_status_clicked)
    blogs.on_click(on_logs_clicked)
    bfailed.on_click(on_failed_clicked)

    d_columns.observe(on_table_clicked)
    d_score_columns.observe(on_table_clicked)

    # meta stuff and column filtration
    def on_bmeta_clicked(b):
        db.vars['show_meta'] = 1 - db.vars.get('show_meta', 0)
        on_table_clicked(None)

    def on_hparam_diff_clicked(b):
        db.vars['hparam_diff'] = 2 - db.vars.get('hparam_diff', 0)
        on_table_clicked(None)

    b_meta.on_click(on_bmeta_clicked)
    b_diff.on_click(on_hparam_diff_clicked)


def columns_widget(label, column_list):
    names = []
    checkbox_objects = []
    objects = [label]
    for key in column_list:
        c = widgets.Checkbox(value=False)
        checkbox_objects.append(c)
        l = widgets.Label(key)
        l.layout.width='500ex'
        objects.append(c)
        objects.append(l)
        names.append(key)

    arg_dict = {names[i]: checkbox for i, checkbox in enumerate(checkbox_objects)}
    ui = widgets.HBox(children=objects)
    selected_data = []

    def select_data(**kwargs):
        selected_data.clear()

        for key in kwargs:
            if kwargs[key] is True:
                selected_data.append(key)

        print(selected_data)

    # out = widgets.interactive_output(select_data, arg_dict)
    return ui, arg_dict