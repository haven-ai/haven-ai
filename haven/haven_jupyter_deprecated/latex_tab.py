from . import widgets as wdg


try:
    from ipywidgets import widgets

    from IPython.display import display
    from IPython.core.display import display
except Exception:
    pass


def latex_tab(db, output):

    b_table = widgets.Button(description="Display Latex Table")

    w_rows = wdg.SelectMultiple(header="Legend", options=db.rm.exp_params, db_vars=db.vars, var="latex_rows")
    w_cols = wdg.SelectMultiple(header="Metrics", options=db.rm.score_keys, db_vars=db.vars, var="latex_columns")

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

            score_table = db.rm.get_latex_table(legend=w_cols.update(), metrics=w_rows.update(), caption="Results")
            print(score_table)

    b_table.on_click(on_clicked)
