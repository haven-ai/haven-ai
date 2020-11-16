import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import wandb
from haven import haven_wizard as hw


def filter_exp_list(exp_list, filterby_list, savedir_base=None, verbose=True,
                    score_list_name='score_list.pkl', return_style_list=False):
    """[summary]
    
    Parameters
    ----------
    exp_list : [type]
        A list of experiments, each defines a single set of hyper-parameters
    filterby_list : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    if filterby_list is None or filterby_list == '' or len(filterby_list) == 0:
        if return_style_list:
            return exp_list, [{}]*len(exp_list)
        else:
            return exp_list

    style_list = []
    filterby_list_list = hu.as_double_list(filterby_list)
    # filterby_list = filterby_list_list
    
    for filterby_list in filterby_list_list:
        exp_list_new = []

        # those with meta
        filterby_list_no_best = []
        for filterby_dict in filterby_list:
            meta_dict = {}
            if isinstance(filterby_dict, tuple):
                fd, meta_dict = filterby_dict
            
            if meta_dict.get('best'):
                assert savedir_base is not None
                el = filter_exp_list(exp_list, filterby_list=fd, verbose=verbose)
                best_dict = meta_dict.get('best')
                exp_dict = get_best_exp_dict(el, savedir_base, 
                                metric=best_dict['metric'],
                                metric_agg=best_dict['metric_agg'], 
                                filterby_list=None, 
                                avg_across=best_dict.get('avg_across'),
                                return_scores=False, 
                                verbose=verbose,
                                score_list_name=score_list_name)

                exp_list_new += [exp_dict]
                style_list += [meta_dict.get('style', {})]
            else:
                filterby_list_no_best += [filterby_dict] 

        
        # ignore metas here meta
        for exp_dict in exp_list:
            select_flag = False
            
            for fd in filterby_list_no_best:
                if isinstance(fd, tuple):
                    filterby_dict, meta_dict = fd
                    style_dict = meta_dict.get('style', {})
                else:
                    filterby_dict = fd
                    style_dict = {}

                filterby_dict = copy.deepcopy(filterby_dict)
               
                keys = filterby_dict.keys()
                for k in keys:
                    if '.' in k:
                        v = filterby_dict[k]
                        k_list = k.split('.')
                        nk = len(k_list)

                        dict_tree = dict()
                        t = dict_tree

                        for i in range(nk):
                            ki = k_list[i]
                            if i == (nk - 1):
                                t = t.setdefault(ki, v)
                            else:
                                t = t.setdefault(ki, {})

                        filterby_dict = dict_tree

                assert (isinstance(filterby_dict, dict), 
                                 ('filterby_dict: %s is not a dict' % str(filterby_dict)))

                if hu.is_subset(filterby_dict, exp_dict):
                    select_flag = True
                    break

            if select_flag:
                exp_list_new += [exp_dict]
                style_list += [style_dict]

        exp_list = exp_list_new
        

    if verbose:
        print('Filtered: %d/%d experiments gathered...' % (len(exp_list_new), len(exp_list)))
    # hu.check_duplicates(exp_list_new)
    exp_list_new = hu.ignore_duplicates(exp_list_new)
    
    if return_style_list:
        return exp_list_new, style_list

    return exp_list_new
  
if __name__ == "__main__":
  exp_list = [{'dataset':'mnist', 'model':{'name':'lenet', 'lr':1e-3}},
              {'dataset':'mnist', 'model':{'name':'kale', 'lr':1e-5}}
              ]
   
  # this works
  filterby_list = [{'model':{lr':1e-5, 'name':'kale'}}]
  
  # this might not work
  filterby_list = [{'model.lr':1e-5, 'model.name':'kale'}]
  
  # https://github.com/haven-ai/haven-ai/blob/master/haven/haven_utils/exp_utils.py#L361
  exp_list_filtered = filter_exp_list(exp_list, filterby_list)
  
  print(exp_list_filtered)
