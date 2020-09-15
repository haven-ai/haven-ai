import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np

from src import models
from src import datasets

import argparse

from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from baal.active.dataset import ActiveLearningDataset

cudnn.benchmark = True

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc


def trainval(exp_dict, savedir_base, datadir_base, reset=False):
    # bookkeeping stuff
    # ==================
    pprint.pprint(exp_dict)
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    if reset:
        hc.delete_and_backup_experiment(savedir)
    
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    print("Experiment saved in %s" % savedir)

    # Dataset
    # ==================

    # load train and acrtive set
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split="train",
                                     datadir_base=datadir_base,
                                     exp_dict=exp_dict)

    active_set = ActiveLearningDataset(train_set, random_state=42) 
    
    # val set
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], 
                                   split="val",
                                   datadir_base=datadir_base,
                                    exp_dict=exp_dict)
    val_loader = DataLoader(val_set, 
                            batch_size=exp_dict["batch_size"])

    # Model
    # ==================
    model = models.get_model(model_name=exp_dict['model']['name'], 
                             exp_dict=exp_dict).cuda()


    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.set_state_dict(hu.torch_load(model_path))
        active_set.load_state_dict(hu.load_pkl(os.path.join(savedir, "active_set.pkl"))) 
        score_list = hu.load_pkl(score_list_path)
        inner_s_epoch = score_list[-1]['inner_epoch'] + 1
        s_cycle = score_list[-1]['cycle']
    else:
        # restart experiment
        score_list = []
        inner_s_epoch = 0
        s_cycle = 0
        
    
    # Train & Val
    # ==================
    print("Starting experiment at cycle %d epoch %d" % (s_cycle, inner_s_epoch))

    for c in range(s_cycle, exp_dict['max_cycle']):
        # Set seed
        np.random.seed(c)
        torch.manual_seed(c)
        torch.cuda.manual_seed_all(c)

        if inner_s_epoch == 0:
            active_set.label_next_batch(model)
            hu.save_pkl(os.path.join(savedir, "active_set.pkl"), active_set.state_dict())

        train_loader = DataLoader(active_set, 
                              sampler=samplers.get_sampler(exp_dict['sampler']['train'], active_set),
                              batch_size=exp_dict["batch_size"])
        # Visualize the model
        model.vis_on_loader(vis_loader, savedir=os.path.join(savedir, "images"))
        
        for e in range(inner_s_epoch, exp_dict['max_epoch']):
            # Validate only at the start of each cycle
            score_dict = {}   
            if e == 0:
                score_dict.update(model.val_on_loader(val_loader))

            # Train the model
            score_dict.update(model.train_on_loader(train_loader))

            # Validate the model
            score_dict["epoch"] = len(score_list)
            score_dict["inner_epoch"] = e
            score_dict["cycle"] = c
            score_dict['n_ratio'] = active_set.n_labelled_ratio
            score_dict["n_train"] = len(train_loader.dataset)
            score_dict["n_pool"] = len(train_loader.dataset.pool)

            # Add to score_list and save checkpoint
            score_list += [score_dict]

            # Report & Save
            score_df = pd.DataFrame(score_list)
            print("\n", score_df.tail(), "\n")
            hu.torch_save(model_path, model.get_state_dict())
            hu.save_pkl(score_list_path, score_list)
            print("Checkpoint Saved: %s" % savedir)   
        
        inner_s_epoch = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir_base', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", default=None)

    args = parser.parse_args()

    # Collect experiments
    # =====================
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Run experiments or View them
    # =====================
    if args.view_experiments:
        # view experiments
        hr.create_jupyter(fname='results/notebook.ipynb', savedir_base=args.savedir_base)

    elif args.run_jobs:
        import user_config
        
        run_command = ('python trainval.py -ei <exp_id> -sb %s' %  (args.savedir_base))
        hjb.run_exp_list_jobs(exp_list,
                            savedir_base=args.savedir_base,
                            workdir=os.path.dirname(os.path.realpath(__file__)),
                            run_command=run_command,
                            job_config=user_config.JOB_CONFIG)

    else:
        # run experiments one by one
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    datadir_base=args.datadir_base,
                    reset=args.reset)

