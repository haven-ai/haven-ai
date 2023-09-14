from .. import haven_utils as hu

import os, glob
import pprint

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
                    out = hu.load_json(glob.glob(path)[0])
                    text_list += [out]
                else:
                    text_list += ["None"]

            n_logs = len(text_list)

            for i, (exp_dict, text) in enumerate(zip(db.rm.exp_list, text_list)):
                exp_id = hu.hash_dict(exp_dict)
                savedir = os.path.join(db.rm_original.savedir_base, exp_id)
                print("\nPredictions %d/%d" % (i + 1, n_logs), "=" * 50)
                print("exp_id:", exp_id)
                print("savedir:", savedir)

                print("\nexp_dict")
                print("-" * 50)
                pprint.pprint(exp_dict)

                if text is not "None":
                    print("\nscore_dict")
                    print("-" * 50)
                    score_list = hu.load_pkl(os.path.join(savedir, "score_list.pkl"))
                    pprint.pprint(score_list[-1])

                print("\nPredictions")
                print("-" * 50)
                for j, t in enumerate(text):
                    print(f"\n*** Example {j} ***\n")
                    pprint.pprint(t)

    blogs.on_click(on_logs_clicked)
