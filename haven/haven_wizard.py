import os, argparse
import pandas as pd

from . import haven_utils as hu 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', default=None)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)

    args = parser.parse_args()

    return args

def run_wizard(func, exp_list=None, exp_groups=None, job_config=None, account_id=None):
    args = get_args()

    # Collect experiments
    # ===================
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))

        exp_list = [exp_dict]

    elif exp_list is None:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_groups[exp_group_name]

    # Run experiments
    # ===============
    if not args.run_jobs:
        for exp_dict in exp_list:
            savedir = create_experiment(exp_dict, args.savedir_base, reset=args.reset, 
                                        verbose=True)
            # do trainval
            func(exp_dict=exp_dict,
                 savedir=savedir)
    else:
        # launch jobs
        from haven import haven_jobs as hjb
        
        jm = hjb.JobManager(exp_list=exp_list, 
                    savedir_base=args.savedir_base, 
                    account_id=account_id,
                    workdir=os.getcwd(),
                    job_config=job_config,
                    )

        command = ('python trainval.py -ei <exp_id> -sb %s' %  
                  (args.savedir_base))

        print(command)
        jm.launch_menu(command=command)


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
    if not os.path.exists(exp_dict_json_fname):
        hu.save_json(exp_dict_json_fname, exp_dict)
    
    #-- score_list
    score_list_fname = os.path.join(savedir, "score_list.pkl")
    if not os.path.exists(score_list_fname):
        hu.save_pkl(score_list_fname, [])
        
    #-- model
    model_fname = os.path.join(savedir, "model.pth")
    if not os.path.exists(model_fname):
        hu.torch_save(model_fname, {})

    #-- images
    os.makedirs(os.path.join(savedir, 'images'), exist_ok=True)

    if copy_code:
        src = os.getcwd() + "/"
        dst = os.path.join(savedir, 'code')
        hu.copy_code(src, dst)

    if verbose:
        pprint.pprint(exp_dict)
        print("> Experiment saved in %s\n" % savedir)

    if return_exp_id:
        return savedir, exp_id

    return savedir

def save_checkpoint(savedir, score_list, model_state_dict=None, 
                    images=None, images_fname=None, fname_suffix='', verbose=True):
    # Report
    if verbose:
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")

    print('Saving in %s' % savedir)
    # save score_list
    score_list_fname = os.path.join(savedir, 'score_list%s.pkl' % fname_suffix)
    hu.save_pkl(score_list_fname, score_list)
    if verbose:
        print('> Saved "score_list" as %s' % os.path.split(score_list_fname)[-1])

    # save model
    if model_state_dict is not None:
        model_state_dict_fname = os.path.join(savedir, 'model%s.pth'% fname_suffix)
        hu.torch_save(model_state_dict_fname, model_state_dict)
        if verbose:
            print('> Saved "model_state_dict" as %s' % os.path.split(model_state_dict_fname)[-1])

    # save images
    images_dir = os.path.join(savedir, 'images%s'% fname_suffix)
    if images is not None:
        for i, img in enumerate(images):
            hu.save_image(os.path.join(images_dir, '%d.png' % i), img)
        if verbose:
            print('> Saved "images" in %s' % os.path.split(images_dir)[-1])


def get_checkpoint(savedir, return_model_state_dict=False):
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

