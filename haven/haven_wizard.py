import os
import argparse
import pandas as pd
import pprint

from . import haven_utils as hu


def get_args():
    parser = argparse.ArgumentParser(formatter_class=make_wide(argparse.ArgumentDefaultsHelpFormatter))

    parser.add_argument('-e', '--exp_group_list', nargs="+",
                        help='Define which exp groups to run.')
    parser.add_argument('-sb', '--savedir_base', default=None,
                        help='Define the base directory where the experiments will be saved.')
    parser.add_argument('-d', '--datadir', default=None,
                        help='Define the dataset directory.')
    parser.add_argument("-r", "--reset",  default=0, type=int,
                        help='Reset or resume the experiment.')
    parser.add_argument("-ei", "--exp_id", default=None,
                        help='Run a specific experiment based on its id.')
    parser.add_argument("-j", "--run_jobs", default=0, type=int,
                        help='Run the experiments as jobs in the cluster.')
    parser.add_argument("-nw", "--num_workers", type=int, default=0,
                        help='Specify the number of workers in the dataloader.')
    parser.add_argument("-v", "--visualize_notebook", type=str, default='',
                        help='Create a jupyter file to visualize the results.')

    args, others = parser.parse_known_args()

    return args

def make_wide(formatter, w=120, h=36):
    """Return a wider HelpFormatter, if possible."""
    try:
        # https://stackoverflow.com/a/5464440
        # beware: "Only the name of this class is considered a public API."
        kwargs = {'width': w, 'max_help_position': h}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        import warnings
        warnings.warn("argparse help formatter failed, falling back.")
        return formatter

def run_wizard(func, exp_list=None, exp_groups=None, job_config=None, 
                savedir_base=None, 
               reset=None, args=None, use_threads=True,
               exp_id=None):
    if args is None:
        args = get_args()

    # Asserts
    # =======
    savedir_base = savedir_base or args.savedir_base
    reset = reset or args.reset
    exp_id = exp_id or args.exp_id
    assert savedir_base is not None

    # Collect experiments
    # ===================
    if exp_id is not None:
        # select one experiment
        savedir = os.path.join(savedir_base, exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    elif exp_list is None:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_groups[exp_group_name]

    # save results folder
    if exp_id is None:
        results_fname = args.visualize_notebook
        if len(results_fname):
            if '.ipynb' not in results_fname:
                results_fname += '.ipynb'
            create_jupyter_file(fname=results_fname,
                                savedir_base=savedir_base)

    hu.check_duplicates(exp_list)
    print('\nRunning %d experiments' % len(exp_list))

    # Run experiments
    # ===============
    if not args.run_jobs:
        for exp_dict in exp_list:
            savedir = create_experiment(exp_dict, savedir_base, reset=reset,
                                        verbose=True)
            # do trainval
            func(exp_dict=exp_dict,
                 savedir=savedir,
                 args=args)
    else:
        # launch jobs
        from haven import haven_jobs as hjb
        assert job_config is not None
        assert 'account_id' in job_config
        jm = hjb.JobManager(exp_list=exp_list,
                            savedir_base=savedir_base,
                            workdir=os.getcwd(),
                            job_config=job_config,
                            )

        command = ('python trainval.py -ei <exp_id> -sb %s -d %s' %
                   (savedir_base, args.datadir))

        print(command)
        jm.launch_menu(command=command, in_parallel=use_threads)


def create_jupyter_file(fname, savedir_base):
    if not os.path.exists(fname):
        cells = [main_cell(savedir_base)]
        save_ipynb(fname, cells)
        print('> Open %s to visualize results' % fname)


def save_ipynb(fname, script_list):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_code_cell(code) for code in
                   script_list]
    with open(fname, 'w') as f:
        nbf.write(nb, f)


def main_cell(savedir_base):
    script = ("""
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu

# path to where the experiments got saved
savedir_base = '%s'
exp_list = None

# filter exps
# e.g. filterby_list =[{'dataset':'mnist'}] gets exps with mnist
filterby_list = None

# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      filterby_list=filterby_list,
                      verbose=0,
                      exp_groups=None
                     )

# launch dashboard
# make sure you have 'widgetsnbextension' enabled; 
# otherwise see README.md in https://github.com/haven-ai/haven-ai

hj.get_dashboard(rm, vars(), wide_display=False, enable_datatables=False)
          """ % savedir_base)
    return script


def create_experiment(exp_dict, savedir_base, reset, copy_code=False, return_exp_id=False, verbose=True):
    import pprint
    from . import haven_chk as hc

    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        hc.delete_and_backup_experiment(savedir)

    # create experiment structure
    os.makedirs(savedir, exist_ok=True)

    #-- exp_dict
    exp_dict_json_fname = os.path.join(savedir, "exp_dict.json")
    # if not os.path.exists(exp_dict_json_fname):
    hu.save_json(exp_dict_json_fname, exp_dict)

    #-- images
    os.makedirs(os.path.join(savedir, 'images'), exist_ok=True)

    if copy_code:
        src = os.getcwd() + "/"
        dst = os.path.join(savedir, 'code')
        hu.copy_code(src, dst)

    if verbose:
        print('\n******')
        print('Exp id: %s' % exp_id)
        print('\nHyperparameters:\n' + "-"*16)
        # print(pd.DataFrame([exp_dict]).to_string(index=False))
        pprint.pprint(exp_dict)

        print("\nSave directory: %s" % savedir)
        print("=" * 100)
        

    if return_exp_id:
        return savedir, exp_id

    return savedir



class Checkpointer:
    def __init__(self, savedir, return_model_state_dict=False, verbose=True):
        self.savedir = savedir
        self.verbose = verbose
        self.chk_dict = get_checkpoint(savedir, return_model_state_dict=return_model_state_dict)

    def save_checkpoint(self, score_dict, score_list=None, model_state_dict=None, images=None, 
                        images_fname=None, fname_suffix=''):
        
        save_checkpoint(savedir, score_list, model_state_dict=model_state_dict,
                    images=images, images_fname=images_fname, 
                    fname_suffix=fname_suffix, verbose=self.verbose)
    
def save_checkpoint(savedir, score_list, model_state_dict=None,
                    images=None, images_fname=None, fname_suffix='', verbose=True):
    # Report
    if verbose:
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))
        
        print('\nExp id: %s' % hu.hash_dict(exp_dict))

        print('\nHyperparameters:\n' + "-"*16)
        # print(pd.DataFrame([exp_dict]).to_string(index=False))
        pprint.pprint(exp_dict)
        print('\nMetrics:\n' + "-"*8)
        score_df = pd.DataFrame(score_list)
        print(score_df.tail().to_string(index=False), "\n")

        
        print("Save directory: %s" % savedir)

    # save score_list
    score_list_fname = os.path.join(savedir, 'score_list%s.pkl' % fname_suffix)
    hu.save_pkl(score_list_fname, score_list)
    # if verbose:
        # print('> Saved "score_list" as %s' %
        #       os.path.split(score_list_fname)[-1])

    # save model
    if model_state_dict is not None:
        model_state_dict_fname = os.path.join(
            savedir, 'model%s.pth' % fname_suffix)
        hu.torch_save(model_state_dict_fname, model_state_dict)
        # if verbose:
            # print('> Saved "model_state_dict" as %s' %
            #       os.path.split(model_state_dict_fname)[-1])

    # save images
    images_dir = os.path.join(savedir, 'images%s' % fname_suffix)
    if images is not None:
        for i, img in enumerate(images):

            if images_fname is not None:
                fname = '%s' % images_fname[i]
            else:
                fname = '%d.png' % i
            hu.save_image(os.path.join(images_dir, fname), img)
        # if verbose:
        #     print('> Saved "images" in %s' % os.path.split(images_dir)[-1])

    print("=" * 100)

def get_checkpoint(savedir, return_model_state_dict=False):
    chk_dict = {}

    # score list
    score_list_fname = os.path.join(savedir, 'score_list.pkl')
    if os.path.exists(score_list_fname):
        score_list = hu.load_pkl(score_list_fname)
    else:
        score_list = []

    chk_dict['score_list'] = score_list
    if len(score_list) == 0:
        chk_dict['epoch'] = 0
    else:
        chk_dict['epoch'] = score_list[-1]['epoch'] + 1

    model_state_dict_fname = os.path.join(savedir, 'model.pth')
    if return_model_state_dict:
        if os.path.exists(model_state_dict_fname):
            chk_dict['model_state_dict'] = hu.torch_load(
                model_state_dict_fname)
        else:
            chk_dict['model_state_dict'] = {}

    return chk_dict


def create_jupyter(savedir, return_model_state_dict=False):
    chk_dict = {}

    # score list
    score_list_fname = os.path.join(savedir, 'score_list.pkl')
    score_list = hu.load_pkl(score_list_fname)

    chk_dict['score_list'] = score_list
    if len(score_list) == 0:
        chk_dict['epoch'] = 0
    else:
        chk_dict['epoch'] = score_list[-1]['epoch'] + 1

    if return_model_state_dict:
        model_state_dict_fname = os.path.join(savedir, 'model.pth')
        chk_dict['model_state_dict'] = hu.torch_load(model_state_dict_fname)

    return chk_dict
