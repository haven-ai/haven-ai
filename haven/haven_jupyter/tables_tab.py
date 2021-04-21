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


def tables_tab(db, output):
    if db.vars.get("legend_list") is None:
        db.vars["legend_list"] = hu.get_diff_hparam(db.rm.exp_list)
        
    w_columns = wdg.SelectMultiple(
        header="Legend:",
        options=["exp_id"] + db.rm.exp_params,
        db_vars=db.vars,
        var="legend_list",
        select_all=False,
    )
    w_score_columns = wdg.SelectMultiple(
        header="Metrics:", options=db.rm.score_keys, db_vars=db.vars, var="score_columns", select_all=False
    )

    bstatus = widgets.Button(description="Jobs Status")
    blogs = widgets.Button(description="Jobs Logs")
    bfailed = widgets.Button(description="Jobs Failed")

    b_table = widgets.Button(description="Display Table")
    b_meta = widgets.Button(description="Display Meta Table")
    b_diff = widgets.Button(description="Display Filtered Table")

    # download logs
    bdownload = widgets.Button(description="Download Logs")
    bdownload_out = widgets.Output()

    w_avg_across = wdg.Dropdown(
        header="Avg Across", options=["None"] + db.rm.exp_params, db_vars=db.vars, var="avg_across"
    )

    button = widgets.VBox(
        [
            widgets.HBox([w_columns.get_widget(), w_score_columns.get_widget(), w_avg_across.get_widget()]),
            widgets.HBox([b_table, bstatus, blogs, bfailed, bdownload, bdownload_out]),
        ]
    )
    output_plot = widgets.Output()

    with output:
        display(button)
        display(output_plot)

    def on_table_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()

            score_table = db.rm.get_score_table(
                columns=w_columns.update(), score_columns=w_score_columns.update(), avg_across=w_avg_across.update()
            )
            display(score_table)

    def on_job_status_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()
            summary_list = db.rm.get_job_summary(
                verbose=db.rm.verbose, add_prefix=True, job_scheduler=db.rm.job_scheduler
            )
            summary_dict = hu.group_list(summary_list, key="job_state", return_count=True)
            display(summary_dict)

            summary_dict = hu.group_list(summary_list, key="job_state", return_count=False)

            for state in summary_dict:
                n_jobs = len(summary_dict[state])
                if n_jobs:
                    df = pd.DataFrame(summary_dict[state])
                    df_head = df.head()
                    display(f"Experiments {state}: {n_jobs} - Only {len(df_head)} experiments are shown.")

                    display(df_head)

    def on_logs_clicked(b):
        output_plot.clear_output()
        with output_plot:
            summary_list = db.rm.get_job_summary(
                verbose=db.rm.verbose, add_prefix=True, job_scheduler=db.rm.job_scheduler
            )

            n_logs = len(summary_list)

            for i, logs in enumerate(summary_list):
                print("\nLogs %d/%d" % (i + 1, n_logs), "=" * 50)
                print("exp_id:", logs["exp_id"])
                print("job_id:", logs["job_id"])
                print("job_state:", logs["job_state"])
                print("command:", logs["command"])
                print("savedir:", os.path.join(db.rm_original.savedir_base, logs["exp_id"]))

                print("\nexp_dict")
                print("-" * 50)
                pprint.pprint(logs["exp_dict"])

                print("\nLogs")
                print("-" * 50)
                pprint.pprint(logs.get("logs"))

    def get_logs(failed_only=False):
        summary_list = db.rm.get_job_summary(verbose=db.rm.verbose, add_prefix=True, job_scheduler=db.rm.job_scheduler)
        summary_dict = hu.group_list(summary_list, key="job_state", return_count=False)
        if "FAILED" not in summary_dict:
            stdout = "NO FAILED JOBS"
            return stdout

        n_failed = len(summary_dict["FAILED"])

        if n_failed == 0:
            stdout = "no failed experiments\n"
        else:
            stdout = ""
            for i, failed in enumerate(summary_dict["FAILED"]):
                stdout += "\nFailed %d/%d " % (i + 1, n_failed) + "=" * 50
                stdout += "\nexp_id: " + failed["exp_id"]
                stdout += "\njob_id: " + failed["job_id"]
                stdout += "\njob_state: " + "FAILED"
                stdout += "\ncommand:" + failed.get("command")
                stdout += "\nsavedir: " + os.path.join(db.rm_original.savedir_base, failed["exp_id"])

                stdout += "\n\nexp_dict"
                stdout += "\n" + "-" * 50 + "\n"
                stdout += pprint.pformat(failed["exp_dict"])

                stdout += "\n\nLogs\n"
                stdout += "-" * 50 + "\n"
                stdout += pprint.pformat(failed.get("logs"))
                stdout += "\n"

        return stdout

    def on_failed_clicked(b):
        output_plot.clear_output()
        with output_plot:
            db.update_rm()
            stdout = get_logs(failed_only=True)
            print(stdout)

    def on_download_clicked(b):
        fname = "logs.txt"
        hu.save_txt(fname, get_logs(failed_only=True))

        bdownload_out.clear_output()

        with bdownload_out:
            display(FileLink(fname, result_html_prefix="Download: "))

    bdownload.on_click(on_download_clicked)

    # Add call listeners
    b_table.on_click(on_table_clicked)
    bstatus.on_click(on_job_status_clicked)
    blogs.on_click(on_logs_clicked)
    bfailed.on_click(on_failed_clicked)
