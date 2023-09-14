import numpy as np

import pandas as pd


def get_latex_table(
    score_df,
    legend=None,
    metrics=None,
    filter_dict=dict(),
    map_row_dict_dict=dict(),
    map_col_dict=dict(),
    decimals=None,
    transposed=True,
    **kwargs
):
    columns = legend
    rows = metrics
    # break it
    dicts = score_df.T.to_dict()

    dicts_new = {}
    for i in dicts:
        exp_score_dict = dicts[i]
        legend_label = "+".join(
            [str(map_row_dict_dict.get(exp_score_dict[l], exp_score_dict[l])) for l in exp_score_dict if l in rows]
        )
        metric_scores = {}
        for k, v in exp_score_dict.items():
            if k in columns:
                if decimals is not None:
                    v = np.round(v, decimals)
                metric_scores[k] = v
        dicts_new[legend_label] = metric_scores

    df_new = pd.DataFrame(dicts_new)
    if transposed:
        df_new = df_new.T

    return df_new.to_latex(**kwargs)
