from .. import haven_results as hr
from .. import haven_utils as hu

from .share_tab import share_tab
from .plots_tab import plots_tab
from .tables_tab import tables_tab
from .latex_tab import latex_tab
from .images_tab import images_tab
from .text_tab import text_tab

import haven
import os

try:
    import ast
    from ipywidgets import Button, HBox, VBox
    from ipywidgets import widgets

    from IPython.display import display
    from IPython.core.display import Javascript, display, HTML
    from IPython.display import FileLink, FileLinks
    from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
except Exception:
    pass


def get_dashboard(rm, vars=None, show_jobs=True, wide_display=False, enable_datatables=True):
    dm = DashboardManager(
        rm, vars=vars, show_jobs=show_jobs, wide_display=wide_display, enable_datatables=enable_datatables
    )
    dm.display()
    return dm


class DashboardManager:
    def __init__(self, rm, vars=None, show_jobs=True, wide_display=True, enable_datatables=True):
        self.rm_original = rm
        if vars is None:
            self.vars = {}
        else:
            self.vars = vars

        self.show_jobs = show_jobs
        self.wide_display = wide_display
        self.enable_datatables = enable_datatables

        self.layout = widgets.Layout(width="100px")
        self.layout_label = widgets.Layout(width="200px")
        self.layout_dropdown = widgets.Layout(width="200px")
        self.layout_button = widgets.Layout(width="200px")
        self.t_savedir_base = widgets.Text(
            value=str(self.vars.get("savedir_base") or rm.savedir_base),
            layout=widgets.Layout(width="600px"),
            disabled=False,
        )

        self.t_filterby_list = widgets.Text(
            value=str(self.vars.get("filterby_list")),
            layout=widgets.Layout(width="1200px"),
            description="               filterby_list:",
            disabled=False,
        )

    def display(self):
        self.update_rm(display_meta=False)

        header = widgets.Label(
            value="Loading Dashboard...",
            layout=widgets.Layout(width="800px"),
        )
        display(header)

        if self.enable_datatables:
            init_datatable_mode()
        tables = widgets.Output()
        plots = widgets.Output()
        images = widgets.Output()
        text = widgets.Output()
        share = widgets.Output()
        latex = widgets.Output()

        main_out = widgets.Output()
        # Display tabs
        tab = widgets.Tab(children=[tables, plots, images, text, latex, share])
        tab.set_title(1, "Plots")
        tab.set_title(0, "Tables")
        tab.set_title(2, "Images")
        tab.set_title(3, "Text")
        tab.set_title(4, "Latex")
        tab.set_title(5, "Share")

        with main_out:
            display(tab)
            tables.clear_output()
            plots.clear_output()
            images.clear_output()
            text.clear_output()
            latex.clear_output()
            share.clear_output()

            # show tabs
            tables_tab(self, tables)
            plots_tab(self, plots)
            images_tab(self, images)
            text_tab(self, text)
            latex_tab(self, latex)
            share_tab(self, share)

            header.value = (
                f"Dashboard loaded (ver: {haven.__version__}). "
                + f'{len(self.rm_original.exp_list_all)} experiments selected from "{self.rm_original.savedir_base}"'
            )

        display(main_out)

        if self.wide_display:
            display(HTML("<style>.container { width:100% !important; }</style>"))

        # This makes cell show full height display
        style = """
        <style>
            .output_scroll {
                height: unset !important;
                border-radius: unset !important;
                -webkit-box-shadow: unset !important;
                box-shadow: unset !important;
            }
        </style>
        """
        display(HTML(style))

    def update_rm(self, display_meta=True):
        self.rm = hr.ResultManager(
            exp_list=self.rm_original.exp_list_all,
            savedir_base=str(self.t_savedir_base.value),
            filterby_list=hu.get_dict_from_str(str(self.t_filterby_list.value)),
            verbose=self.rm_original.verbose,
            mode_key=self.rm_original.mode_key,
            has_score_list=self.rm_original.has_score_list,
            score_list_name=self.rm_original.score_list_name,
            job_scheduler=self.rm_original.job_scheduler,
        )

        # if len(self.rm.exp_list) == 0:
        #     if self.rm.n_exp_all > 0:
        #         display('No experiments selected out of %d '
        #                 'for filtrby_list %s' % (self.rm.n_exp_all,
        #                                          self.rm.filterby_list))
        #         display('Table below shows all experiments.')
        #         score_table = hr.get_score_df(exp_list=self.rm_original.exp_list_all,
        #                                       savedir_base=self.rm_original.savedir_base)
        #         display(score_table)
        #     else:
        #         display('No experiments exist...')
        #     return
        # else:
        if display_meta:
            display(f'{len(self.rm.exp_list)}/{len(self.rm.exp_list_all)} experiments selected using "filterby_list"')


def launch_jupyter():
    """
    virtualenv -p python3 .
    source bin/activate
    pip install jupyter notebook
    jupyter notebook --ip 0.0.0.0 --port 2222 --NotebookApp.token='abcdefg'
    """
    print()


def create_jupyter_file(
    fname,
    savedir_base,
    overwrite=False,
    print_url=False,
):
    savedir_base = os.path.abspath(savedir_base)
    if overwrite or not os.path.exists(fname):
        cells = [main_cell(savedir_base), sub_cell(), install_cell()]
        if os.path.dirname(fname) != "":
            os.makedirs(os.path.dirname(fname), exist_ok=True)
        save_ipynb(fname, cells)
        print("> Open %s to visualize results" % fname)

    if print_url:
        from notebook import notebookapp

        servers = list(notebookapp.list_running_servers())
        hostname = os.uname().nodename

        flag = False
        for i, s in enumerate(servers):
            if s["hostname"] == "localhost":
                continue
            flag = True
            url = "http://%s:%s/" % (hostname, s["port"])
            print("- url:", url)

        if flag == False:
            print("a jupyter server was not found :(")


def sub_cell():
    script = """
# get table 
rm.get_score_df().head()

# get latex 
# print(rm.get_latex_table(legend=['dataset'], metrics=['train_loss'], decimals=1, caption="Results", label='tab:results'))

# get custom plots
fig = rm.get_plot_all(
                # order='metrics_by_groups',
                # avg_across='runs',
                y_metric_list=y_metrics, 
                x_metric=x_metric,
                # legend_fontsize=18,
                # x_fontsize=20,
                # y_fontsize=20,
                # xtick_fontsize=20,
                # ytick_fontsize=20,
                # title_fontsize=24,
                # legend_list=['model], 
                # title_list = ['dataset'], 
                # title_format='Dataset:{}',
                # log_metric_list = ['train_loss'], 
                # groupby_list = ['dataset'],
                # map_ylabel_list=[{'train_loss':'Train loss'}],
                # map_xlabel_list=[{'epoch':'Epoch'}],
                # figsize=(15,5),
                # plot_confidence=False,
                # savedir_plots='%s' % (name)
)
          """
    return script


def main_cell(savedir_base):
    script = (
        """
from haven import haven_results as hr
from haven import haven_utils as hu

# path to where the experiments got saved
savedir_base = '%s'
exp_list = None

# get experiments from the exp config
exp_config_fname = None
if exp_config_fname:
    config = hu.load_py(exp_config_fname)
    exp_list = []
    for exp_group in [
        "example"
                    ]:
        exp_list += config.EXP_GROUPS[exp_group]

# filter exps

filterby_list = None
# filterby_list =[{'dataset':'mnist'}]

# get experiments
rm = hr.ResultManager(exp_list=exp_list,
                      savedir_base=savedir_base,
                      filterby_list=filterby_list,
                      verbose=0,
                      exp_groups=None,
                      job_scheduler='slurm'
                     )

# specify display parameters

# groupby_list = ['dataset']
# title_list = ['dataset']
# legend_list = ['model']
y_metrics = ['train_loss']
x_metric = 'epoch'

# launch dashboard
hj.get_dashboard(rm, vars(), wide_display=False, enable_datatables=False)
          """
        % savedir_base
    )
    return script


def install_cell():
    script = """
    !pip install --upgrade git+https://github.com/haven-ai/haven-ai
          """
    return script


def save_ipynb(fname, script_list):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_code_cell(code) for code in script_list]
    with open(fname, "w") as f:
        nbf.write(nb, f)


def init_datatable_mode():
    """Initialize DataTable mode for pandas DataFrame represenation."""
    import pandas as pd
    from IPython.core.display import display, Javascript

    # configure path to the datatables library using requireJS
    # that way the library will become globally available
    display(
        Javascript(
            """
        require.config({
            paths: {
                DT: '//cdn.datatables.net/1.10.19/js/jquery.dataTables.min',
            }
        });
        $('head').append('<link rel="stylesheet" type="text/css" href="//cdn.datatables.net/1.10.19/css/jquery.dataTables.min.css">');
    """
        )
    )

    def _repr_datatable_(self):
        """Return DataTable representation of pandas DataFrame."""
        # classes for dataframe table (optional)
        classes = ["table", "table-striped", "table-bordered"]

        # create table DOM
        script = f"$(element).html(`{self.to_html(index=True, classes=classes)}`);\n"

        # execute jQuery to turn table into DataTable
        script += """
            require(["DT"], function(DT) {
                $(document).ready( () => {
                    // Turn existing table into datatable
                    $(element).find("table.dataframe").DataTable({"scrollX": true});

                    $('#container').css( 'display', 'block' );
                    table.columns.adjust().draw();

                })
            });
        """

        return script

    pd.DataFrame._repr_javascript_ = _repr_datatable_
