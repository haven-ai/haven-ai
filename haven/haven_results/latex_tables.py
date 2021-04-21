import sys
import os
import pprint
from haven.haven_results import plots_line as pl
from haven import haven_results as hr
from haven import haven_utils as hu
from haven.haven_jobs import slurm_manager as sm

import pprint
import pandas as pd


def get_latex_table(
    score_df, columns=None, rows=None, filter_dict=dict(), map_row_dict_dict=dict(), map_col_dict=dict(), **kwargs
):

    # break it
    dicts = score_df.T.to_dict()

    dicts_new = {}
    for i in dicts:
        exp_score_dict = dicts[i]
        row_label = "+".join(
            [str(map_row_dict_dict.get(exp_score_dict[l], exp_score_dict[l])) for l in exp_score_dict if l in rows]
        )
        col_scores = {k: v for k, v in exp_score_dict.items() if k in columns}
        dicts_new[row_label] = col_scores

    df_new = pd.DataFrame(dicts_new).T
    return df_new.to_latex(**kwargs)
