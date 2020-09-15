
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

def latex_tab():
    pass

def create_latex_table(table, filter_dict, map_row_dict_dict, map_col_dict, **kwargs):
    '''
    Usage
    -----
        map_row_dict_dict = {'model.loss': {"point_loss":'Point Loss', 
                                        "cons_point_loss":'CB Point Loss', 
                                        'joint_cross_entropy':'W-CE (Full Sup.)'}}

        map_col_dict = {'model.loss': 'Loss Function', 
                        'test_dice':'Dice', 
                        'test_iou':'IoU', 
                        'test_prec':'PPV', 
                        'test_recall':'Sens.', 
                        'test_spec':'Spec.'}

        filter_dict = {'Loss Function':['Point Loss', 'CB Point Loss','W-CE (Full Sup.)']}

        caption_dict = {'weakly_covid19_v1_c2':'COVID-19-A',
                        'weakly_covid19_v2_mixed_c2':'COVID-19-B-Mixed',
                        'weakly_covid19_v3_mixed_c2':'COVID-19-C-Mixed',
                        
                        'weakly_covid19_v2_sep_c2':'COVID-19-B-Sep',
                        'weakly_covid19_v3_sep_c2':'COVID-19-C-Sep'}

        for exp_name in ['weakly_covid19_v1_c2', 
                        
                        'weakly_covid19_v2_mixed_c2', 
                        'weakly_covid19_v3_mixed_c2',
                        
                        'weakly_covid19_v2_sep_c2', 
                        'weakly_covid19_v3_sep_c2']:
            rm.exp_list = hr.get_exp_list_from_config([exp_name], exp_config_name)
            table = (rm.get_score_table())
            print(create_latex_table(table=table, 
                                    filter_dict=filter_dict, 
                                    map_row_dict_dict=map_row_dict_dict, 
                                    map_col_dict=map_col_dict,
                                    float_format='%.2f', 
                                    caption=caption_dict[exp_name], 
        #                              label=caption_dict, 
                                    index=False))
                                    
    '''
    # map columns
    table2 = pd.DataFrame()
    for col_old, col_new in map_col_dict.items():
        # map column
        table2[col_new] = table[col_old]
        
        # map rows
        if col_old in map_row_dict_dict:
            map_row_dict = map_row_dict_dict[col_old]
            table2[col_new] = table2[col_new].apply( lambda x: map_row_dict[x.replace("'","")] if x.replace("'","") in map_row_dict else x )
        
    # filter dict
    conds = None
    for k, v in filter_dict.items():
        if not isinstance(v, list):
            v = [v]
#         print(k, v)
        for vi in v:
            cond = table2[k] == vi
            if conds is None:
                conds = cond
            else:
                conds = conds | cond
        
        table2 = table2[conds]
        table2 = table2.set_index(k)
        print(v)
        table2 = table2.reindex(v)
        table2.insert(0, k, table2.index)
        table2 = table2.reset_index(drop=True)
        

    return table2.to_latex(**kwargs)