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
import exp_configs
import time
import numpy as np

from src import models
from src import datasets
from src.datasets import samplers
from src import utils as ut
from haven import haven_wizard as hw

import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader

cudnn.benchmark = True


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    datadir = args.datadir 
    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Dataset
    # ==================
    # train set
    train_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                     split="train",
                                     datadir=datadir,
                                     exp_dict=exp_dict,
                                     dataset_size=exp_dict['dataset_size'])
    # val set
    val_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="val",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])

    # test set
    test_set = datasets.get_dataset(dataset_dict=exp_dict["dataset"],
                                   split="test",
                                   datadir=datadir,
                                   exp_dict=exp_dict,
                                   dataset_size=exp_dict['dataset_size'])


    # val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set,
                            # sampler=val_sampler,
                            batch_size=exp_dict["batch_size"],
                            collate_fn=ut.collate_fn,
                            num_workers=args.num_workers,
                            drop_last=False)

    test_loader = DataLoader(test_set,
                            # sampler=val_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=args.num_workers)

    # Model 
    # ==================
    model = models.get_model(model_dict=exp_dict['model'],
                             exp_dict=exp_dict,
                             train_set=train_set).cuda()

    chk_dict = hw.get_checkpoint(savedir)
    score_list = chk_dict['score_list']

    # Train & Val
    # ==================
    model.waiting = 0
    model.val_score_best = -np.inf
    
    sampler = exp_dict['dataset'].get('sampler', 'random') 
    if sampler == 'random':
        train_sampler = torch.utils.data.RandomSampler(
                                    train_set, replacement=True, 
                                    num_samples=len(val_set))
    elif sampler == 'balanced':
        train_sampler = samplers.BalancedSampler(
                                    train_set, n_samples=len(val_set))
    train_loader = DataLoader(train_set,
                            sampler=train_sampler,
                            collate_fn=ut.collate_fn,
                            batch_size=exp_dict["batch_size"], 
                            drop_last=True, 
                            num_workers=args.num_workers)
    
    for e in range(chk_dict['epoch'], exp_dict['max_epoch']):
        # Validate only at the start of each cycle
        score_dict = {}
        # Train the model
        train_dict = model.train_on_loader(train_loader)

        # Validate the model
        val_dict = model.val_on_loader(val_loader, 
                                       savedir_images=os.path.join(savedir, "images"), n_images=5)
        score_dict.update(val_dict)

        # Get new score_dict
        score_dict.update(train_dict)
        score_dict["epoch"] = e
        score_dict["waiting"] = model.waiting

        model.waiting += 1

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Save Best Checkpoint
        score_df = pd.DataFrame(score_list)
        if score_dict["val_score"] >= model.val_score_best:
            test_dict = model.val_on_loader(test_loader,
                                    savedir_images=os.path.join(savedir, "images"),
                                    n_images=3)  
            score_dict.update(test_dict)

            hu.save_pkl(os.path.join(savedir, "score_list_best.pkl"), score_list)
            # score_df.to_csv(os.path.join(savedir, "score_best_df.csv"))
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                        model.get_state_dict())
            model.waiting = 0
            model.val_score_best = score_dict["val_score"]
            print("Saved Best: %s" % savedir)

        # Report & Save
        hw.save_checkpoint(savedir, score_list=score_list)

        if model.waiting > 100:
            break

    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    import exp_configs
    hw.run_wizard(func=trainval, exp_groups=exp_configs.EXP_GROUPS)
