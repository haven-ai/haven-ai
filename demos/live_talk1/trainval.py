import torch
import pandas as pd
import os
import argparse

from haven import haven_wizard as hw
from haven import haven_utils as hu

from src import datasets, models


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # -- Datasets
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     train_flag=True,
                                     datadir=args.datadir)

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   train_flag=False,
                                   datadir=args.datadir)

    # -- Model
    model = models.Model(exp_dict, device=torch.device('cuda'))

    # -- Train & Val Loop
    score_list = []
    for e in range(0, 50):
        # Compute metrics
        score_dict = {"epoch": e}
        score_dict["train_loss"] = model.val_on_dataset(
            val_set, metric_name='softmax_loss')
        score_dict["val_acc"] = model.val_on_dataset(
            val_set, metric_name='softmax_acc')
        score_list += [score_dict]

        # Train model for one epoch
        model.train_on_dataset(train_set)

        # Visualize
        images = model.vis_on_dataset(
            val_set, fname=os.path.join(savedir, 'images', 'results.png'))

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")
        hu.save_pkl(os.path.join(savedir, 'score_list.pkl'), score_list)
        hu.torch_save(os.path.join(savedir, 'model.pth'), model.state_dict())
        print("Checkpoint Saved: %s" % savedir)

    print('Experiment completed et epoch %d' % e)


if __name__ == "__main__":
    # -- Create Parser
    parser = argparse.ArgumentParser()

    # Exp Arguments
    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument("-ei", "--exp_id", default=None)

    # Savedir Arguments
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)

    # Others
    parser.add_argument("-r", "--reset", default=0, type=int)
    parser.add_argument("-j", "--run_jobs", default=0, type=int)
    parser.add_argument("-v", "--visualize_notebook", type=str, default='')

    args, others = parser.parse_known_args()

    # -- Launch Experiments

    # Get Experiment Groups
    import exp_configs
    exp_groups = exp_configs.EXP_GROUPS
    print('Launching exp_group: %s' % args.exp_group_list)

    if os.path.exists('job_configs.py'):
        import job_configs
        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    # Run Selected Experiments
    hw.run_wizard(func=trainval, exp_groups=exp_groups, args=args, 
                job_config=job_config)
