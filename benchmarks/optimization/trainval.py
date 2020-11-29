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
import job_configs
import time
import numpy as np
from torch.utils.data import RandomSampler, DataLoader

from src import models
from src import datasets
from src import metrics

import argparse

from src import optimizers


def trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=0, use_cuda=False):
    # bookkeeping
    pprint.pprint(exp_dict)  # print the experiment configuration
    exp_id = hu.hash_dict(exp_dict)  # generate a unique id for the experiment
    savedir = os.path.join(savedir_base, exp_id)  # generate a route with the experiment id
    if reset:
        hc.delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)  # create the route to keep the experiment result
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)  # save the experiment config as json
    print("Experiment saved in %s" % savedir)

    # set seed and device
    # ==================
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not available, please run with "-c 0"'  # check if cuda is available 
    else:
        device = 'cpu'

    print('Running on device: %s' % device)

    # Dataset
    # ==================
    # train set 
    # load the dataset for training from the datasets
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"]["name"],
                                     train_flag=True,
                                     datadir=datadir,
                                     exp_dict=exp_dict)

    train_loader = DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              sampler=None,
                              batch_size=exp_dict["batch_size"])

    # val set
    # load the dataset for validation from the datasets
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"]["name"],
                                    train_flag=False,
                                    datadir=datadir,
                                    exp_dict=exp_dict)


    # Model
    # ==================
    model = models.get_model(exp_dict, train_set=train_set).to(device)
    model_path = os.path.join(savedir, "model.pth")  # generate the route to keep the model of the experiment
    
    # Choose loss and metric function
    loss_function = metrics.get_metric_function(exp_dict["loss_func"])

    # Load Optimizer
    # ==============
    n_batches_per_epoch = len(train_set) / float(exp_dict["batch_size"])
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                   params=model.parameters(),
                                   n_batches_per_epoch=n_batches_per_epoch,
                                   n_train=len(train_set),
                                   train_loader=train_loader,
                                   model=model,
                                   loss_function=loss_function, 
                                   exp_dict=exp_dict,
                                   batch_size=exp_dict["batch_size"])
    opt_path = os.path.join(savedir, "opt_state_dict.pth")
    
    score_list_path = os.path.join(savedir, "score_list.pkl")
    if os.path.exists(score_list_path):  
        # resume experiment from the last checkpoint, load the latest model
        # epoch starts from last completed epoch plus one
        model.load_state_dict(hu.torch_load(model_path))
        opt.load_state_dict(hu.torch_load(opt_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        # epoch starts from zero
        score_list = []
        s_epoch = 0

    # Train & Val
    # ==================
    print("Starting experiment at epoch %d" % (s_epoch))

    train_sampler = RandomSampler(data_source=train_set, replacement=True, num_samples=2*len(val_set))
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=exp_dict["batch_size"],
                              drop_last=True, num_workers=num_workers)

    val_sampler = torch.utils.data.SequentialSampler(val_set)
    val_loader = DataLoader(val_set,
                            sampler=val_sampler,
                            batch_size=1,
                            num_workers=num_workers)

    e = s_epoch 
    for e in range(s_epoch, exp_dict["max_epoch"]):
        # Set seed
        seed = e + exp_dict['runs']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        score_dict = {"epoch": e}

        # run with metrics to validate the model
        # 1. Compute train loss over train set
        score_dict["train_loss"] = metrics.compute_metric_on_dataset(model, 
                                            train_set,
                                            metric_name=exp_dict["loss_func"],
                                            batch_size=exp_dict['batch_size'])

        # 2. Compute val acc over val set
        score_dict["val_acc"] = metrics.compute_metric_on_dataset(model, val_set,
                                        metric_name=exp_dict["acc_func"],
                                            batch_size=exp_dict['batch_size'])

        # Train
        # -----
        model.train()
        print("%d - Training model with %s..." % (e, exp_dict["loss_func"]))

        s_time = time.time()

        train_on_loader(model, train_set, train_loader, opt, loss_function, e)
      
        e_time = time.time()

        # Train the model
        score_dict["step"] = opt.state.get("step", 0) / int(n_batches_per_epoch)
        score_dict["step_size"] = opt.state.get("step_size", {})
        score_dict["step_size_avg"] = opt.state.get("step_size_avg", {})
        score_dict["n_forwards"] = opt.state.get("n_forwards", {})
        score_dict["n_backwards"] = opt.state.get("n_backwards", {})
        score_dict["grad_norm"] = opt.state.get("grad_norm", {})
        score_dict["train_epoch_time"] = e_time - s_time
        score_dict.update(opt.state["gv_stats"])

        # Get new score_dict
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")  # print out the epoch, train_loss, and val_acc in the score_list as a table
        hu.torch_save(model_path, model.state_dict()) # save the model state (i.e. state_dic, including optimizer) to the model path
        hu.torch_save(opt_path, opt.state_dict())
        hu.save_pkl(score_list_path, score_list)

        print("Checkpoint Saved: %s" % savedir)

        # Save Best Checkpoint
        if e == 0 or (score_dict.get("val_acc", 0) > score_df["val_acc"][:-1].fillna(0).max()):
            hu.save_pkl(os.path.join(
                savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                          model.state_dict())
            hu.torch_save(os.path.join(savedir, "opt_best.pth"), 
                          opt.state_dict())
            print("Saved Best: %s" % savedir)

    print('Experiment completed et epoch %d' % e)


def train_on_loader(model, train_set, train_loader, opt, loss_function, epoch):
    for batch in tqdm.tqdm(train_loader):
        opt.zero_grad()
        # TODO: change this if optimizer contains more info!! and output of opt_step is not captured?
        optimizers.opt_step(exp_dict['opt']['name'], opt, model, batch, loss_function, False, epoch)


def save_exp_folder(exp_dict, savedir_base, reset):
    exp_id = hu.hash_dict(exp_dict)  # generate a unique id for the experiment
    savedir = os.path.join(savedir_base, exp_id)  # generate a route with the experiment id
    if reset:
        hc.delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)  # create the route to keep the experiment result
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)  # save the experiment config as json


def get_existing_slurm_job_commands(exp_list, savedir_base):
    existing_job_commands = []
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)
        file_name = os.path.join(savedir, "job_dict.json")
        if not os.path.exists(file_name):
            continue
        job_dict = hu.load_json(file_name)
        job_id = job_dict["job_id"]
        job_status = hu.subprocess_call("scontrol show job %s" % job_id).split("JobState=")[1].split(" ")[0]
        if job_status == "RUNNING" or job_status == "PENDING":
            existing_job_commands += [job_dict["command"]]
        
    return existing_job_commands


def launch_slurm_job(command, savedir_base):
    # read slurm setting
    lines = "#! /bin/bash \n"
    lines += "#SBATCH --account=%s \n" % job_configs.ACCOUNT_ID
    for key in list(job_configs.JOB_CONFIG.keys()):
        lines += "#SBATCH --%s=%s \n" % (key, job_configs.JOB_CONFIG[key])
    lines += command

    exp_id = command.split("-ei ")[1].split(" ")[0]  
    file_name = "%s.sh" % exp_id
    hu.save_txt(file_name, lines)
    # launch the exp
    submit_command = "sbatch %s" % file_name
    job_id = hu.subprocess_call(submit_command).split()[-1]

    # save the command and job id in job_dict.json
    job_dict = {
        "command": command,
        "job_id": job_id
    }
    savedir = os.path.join(savedir_base, exp_id)
    hu.save_json(os.path.join(savedir, "job_dict.json"), job_dict)

    # delete the slurm.sh
    os.remove(file_name)


if __name__ == "__main__":
    # create a parser that will hold all the information necessary to parse the command line into Python data type
    parser = argparse.ArgumentParser()

    # add required arguments and default arguments so that the parser know how to take strings on the command line
    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-r", "--reset", default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-s", "--run_slurm", default=0, type=int)
    parser.add_argument("-nw", "--num_workers", type=int, default=0)
    parser.add_argument("-c", "--use_cuda", type=int, default=0)    # user can define whether to run with cuda or cpu, default is not using cuda

    # parse the arguments from the command line
    args = parser.parse_args()

    # Collect experiments
    # ===================
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

    # Run experiments
    # ===============
    # if args.run_jobs:
    #     # run with job scheduler
    #     from haven import haven_jobs as hj
    #     hj.run_exp_list_jobs(exp_list, 
    #                    savedir_base=args.savedir_base, 
    #                    workdir=os.path.dirname(os.path.realpath(__file__)))

    if args.run_slurm:
        # run with job scheduler
        command_list = []
        for exp_dict in exp_list:
            save_exp_folder(exp_dict, args.savedir_base, args.reset)
            exp_id = hu.hash_dict(exp_dict)
            command_list += ["python trainval.py -ei %s -sb %s -d %s -c %d" % (exp_id, args.savedir_base, args.datadir, args.use_cuda)]
        # get slurm existing commands
        existing_commands = get_existing_slurm_job_commands(exp_list, args.savedir_base)
        for command in command_list:
            # check if command exists
            if command in existing_commands:
                print('command exists')
                continue
            # otherwise launch command
            launch_slurm_job(command, args.savedir_base) 
            time.sleep(1)

    else:
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                     savedir_base=args.savedir_base,
                     datadir=args.datadir,
                     reset=args.reset,
                     num_workers=args.num_workers,
                     use_cuda=args.use_cuda)
