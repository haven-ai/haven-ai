import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

from haven import haven_results as hr
from haven import haven_utils as hu
from haven.haven_jobs import slurm_manager as sm

if __name__ == "__main__":
    savedir_base = '.tmp'
    exp_dict = {'model':{'name':'mlp', 'n_layers':30}, 
                'dataset':'mnist', 'batch_size':1}
    exp_dict2 = {'model':{'name':'mlp2', 'n_layers':30}, 
                'dataset':'mnist', 'batch_size':1}

    score_list = [{'epoch': 0, 'acc':0.5}, {'epoch': 0, 'acc':0.9}]

    hu.save_pkl(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                    'score_list.pkl'), score_list)
                    
    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict),
                    'exp_dict.json'), exp_dict)

    hu.save_json(os.path.join(savedir_base, hu.hash_dict(exp_dict2),
                    'exp_dict.json'), exp_dict)
    # check if score_list can be loaded and viewed in pandas
    exp_list = hu.get_exp_list(savedir_base=savedir_base)
    score_df = hr.get_score_df(exp_list, savedir_base=savedir_base)
