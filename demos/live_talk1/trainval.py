from haven import haven_chk as hc
from haven import haven_results as hr
from haven import haven_utils as hu
import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import job_configs
import time
import exp_configs
import job_configs
import numpy as np

from haven import haven_wizard as hw
from torch.utils.data import RandomSampler, DataLoader


import argparse

from src import datasets, models


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # -- datasets
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=args.datadir,
                                     exp_dict=exp_dict)

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                    train_flag=False,
                                    datadir=args.datadir,
                                    exp_dict=exp_dict)

    # -- Model
    model = models.Model(exp_dict, device=torch.device('cuda'))
    
    # -- Train & Val
    score_list = []
    for e in range(0, 50):
        score_dict = {"epoch": e}

        # - Visualize
        images = model.vis_on_dataset(val_set, fname=os.path.join(savedir, 'images', 'results.png'))

        # - Compute metrics
        score_dict["train_loss"] = model.val_on_dataset(val_set, metric_name='softmax_loss')
        score_dict["val_acc"] = model.val_on_dataset(val_set, metric_name='softmax_acc')
        
        # - Train model
        model.train_on_dataset(train_set)
        
        # Get new score_dict
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")  # print out the epoch, train_loss, and val_acc in the score_list as a table
        hu.save_pkl(os.path.join(savedir, 'score_list.pkl'), score_list)
        print("Checkpoint Saved: %s" % savedir)

    print('Experiment completed et epoch %d' % e)

if __name__ == "__main__":
    # create a parser that will hold all the information necessary to parse the command line into Python data type
    parser = argparse.ArgumentParser()

    # add required arguments and default arguments so that the parser know how to take strings on the command line
    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-r", "--reset", default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=0)    # user can define whether to run with cuda or cpu, default is not using cuda
    parser.add_argument("-v", "--visualize_notebook", type=str, default='',
                        help='Create a jupyter file to visualize the results.')
    args, others = parser.parse_known_args()

    # Launch experiments using magic command
    import exp_configs
    
    if os.path.exists('job_configs.py'):
        import job_configs  
        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    hw.run_wizard(func=trainval, exp_groups=exp_configs.EXP_GROUPS,  
                  args=args, job_config=job_config)