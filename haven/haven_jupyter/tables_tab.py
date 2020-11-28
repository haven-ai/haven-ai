
from .. import haven_utils
from .. import haven_results as hr
from .. import haven_utils as hu 
from .. import haven_share as hd

import os
import pprint, json
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
except:
    print('widgets not available...')


def tables_tab(db, output):
    w_columns = wdg.SelectMultiple(header="Hyperparameters:", 
                            options=db.rm.exp_params,
                            db_vars=db.vars, 
                            var='columns')
    w_score_columns = wdg.SelectMultiple(header="Metrics:", 
                            options=db.rm.score_keys,
                            db_vars=db.vars, 
                            var='score_columns')


    bstatus = widgets.Button(description="Jobs Status")
    blogs = widgets.Button(description="Jobs Logs")
    bfailed = widgets.Button(description="Jobs Failed")

    b_table = widgets.Button(description="Display Table")
    b_meta = widgets.Button(description="Display Meta Table")
    b_diff = widgets.Button(description="Display Filtered Table")

    w_avg_across = wdg.Dropdown(header='Avg Across',
                                options=['None'] + db.rm.exp_params,
                            db_vars=db.vars, 
                            var='avg_across')

    button = widgets.VBox([ 
                            widgets.HBox([w_columns.get_widget(), w_score_columns.get_widget(), w_avg_across.get_widget()]),
                            widgets.HBox([b_table, bstatus, blogs, bfailed, ]),
                           
    ])
    output_plot = widgets.Output()

    with output:
        display(button)
        display(output_plot)

    def on_table_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()

            score_table = db.rm.get_score_table(columns=w_columns.update(), 
                                            score_columns=w_score_columns.update(),
                                            avg_across=w_avg_across.update())
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
                pprint.pprint(logs.get('logs'))     
    
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
                    pprint.pprint(failed.get('logs'))

    # Add call listeners
    b_table.on_click(on_table_clicked)
    bstatus.on_click(on_job_status_clicked)
    blogs.on_click(on_logs_clicked)
    bfailed.on_click(on_failed_clicked)



