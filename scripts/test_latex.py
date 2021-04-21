import pandas as pd
import pprint
from haven.haven_jobs import slurm_manager as sm
from haven import haven_utils as hu
from haven import haven_results as hr
from haven.haven_results import plots_line as pl
import sys
import os
import pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)


def get_latex_table(
    score_df, columns=None, rows=None, filter_dict=dict(), map_row_dict_dict=dict(), map_col_dict=dict(), **kwargs
):

    # break it
    dicts = score_df.T.to_dict()

    dicts_new = {}
    for i in dicts:
        exp_score_dict = dicts[i]
        row_label = "_".join(
            [map_row_dict_dict.get(exp_score_dict[l], exp_score_dict[l]) for l in exp_score_dict if l in rows]
        )
        col_scores = {k: v for k, v in exp_score_dict.items() if k in columns}
        dicts_new[row_label] = col_scores

    df_new = pd.DataFrame(dicts_new).T
    return df_new.to_latex(**kwargs)


if __name__ == "__main__":
    savedir_base = ".tmp"
    exp_dict = {"model": {"name": "mlp", "n_layers": 30}, "dataset": "mnist", "batch_size": 1}

    score_list = [{"epoch": 4, "acc": 0.5}, {"epoch": 6, "acc": 0.9}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)

    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)

    exp_dict = {"model": {"name": "mlp2", "n_layers": 35}, "dataset": "mnist", "batch_size": 1}
    score_list = [{"epoch": 2, "acc": 0.1}, {"epoch": 6, "acc": 0.3}]
    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict), "score_list.pkl"), score_list)
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict), "exp_dict.json"), exp_dict)
    # check if score_list can be loaded and viewed in pandas
    exp_list = hu.get_exp_list(savedir_base=savedir_base)
    score_df = hr.get_score_df(exp_list, savedir_base=savedir_base)

    print(
        get_latex_table(
            score_df,
            columns=["acc", "epoch"],
            rows=["model.name"],
            filter_dict=dict(),
            caption="Test",
            float_format="%.2f",
            label="ref:table1",
            map_row_dict_dict=dict(),
            map_col_dict=dict(),
        )
    )
    print()
