# Torch
import torch


# Standard Python libraries
import os
import argparse

# Others
import exp_configs
import src.datasets as datasets
import src.utils as ut
from src import models
from src import dataloaders
import pandas as pd

# External libraries
import pprint

# Haven
from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_img as hi


def train(exp_dict, savedir_base, reset, compute_fid=False):
    # Book keeping
    pprint.pprint(exp_dict)
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    if reset:
        ut.rmtree(savedir)
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    print('Experiment saved in %s' % savedir)

    device = \
        torch.device('cuda:' + exp_dict['gpu'] if torch.cuda.is_available() else 'cpu')

    # 1. Load dataset and loader
    train_set, test_set, num_channels, num_train_classes, num_test_classes = \
        datasets.get_dataset(exp_dict['dataset'],
                             dataset_path=savedir_base,
                             image_size=exp_dict['image_size'])
    train_loader, test_loader = \
            dataloaders.get_dataloader(exp_dict['dataloader'],
                                       train_set, test_set, exp_dict)

    # 2. Fetch model to train
    model = models.get_model(exp_dict['model'],
                             num_train_classes, num_test_classes, 
                             num_channels, device, exp_dict)

    # 3. Resume experiment or start from scratch
    score_list_path = os.path.join(savedir, 'score_list.pkl')
    if os.path.exists(score_list_path):
        # Resume experiment if it exists
        model_path = os.path.join(savedir, 'model_state_dict.pth')
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        meta_dict_path = os.path.join(savedir, 'meta_dict.pkl')
        meta_dict = hu.load_pkl(meta_dict_path)
        print('Resuming experiment at episode %d epoch %d' %
              (meta_dict['episode'], meta_dict['epoch']))
    else:
        # Start experiment from scratch
        meta_dict = {'episode': 1, 'epoch': 1}
        score_list = []

        # Remove TensorBoard logs from previous runs
        ut.rmtree(os.path.join(savedir, 'tensorboard_logs'))

        print('Starting experiment at episode %d epoch %d' %
              (meta_dict['episode'], meta_dict['epoch']))

    # 4. Train and eval loop
    s_epoch = meta_dict['epoch']
    for e in range(s_epoch, exp_dict['num_epochs'] + 1):
        # 0. Initialize dicts 
        score_dict = {'epoch': e}
        meta_dict['epoch'] = e

        # 1. Train on loader 
        train_dict = model.train_on_loader(train_loader)

        # 1b. Compute FID
        if compute_fid == 1:
            if e % 20 == 0 or e == 1 or e == exp_dict['num_epochs']:
                print('Starting FID computation...')
                train_dict['fid'] = fid(model, train_loader.dataset,
                                        train_loader.sampler, save_dir)

        score_dict.update(train_dict)

        # 2. Eval on loader 
        eval_dict = model.val_on_loader(test_loader, savedir, e)
        score_dict.update(eval_dict)
        
        # 3. Report and save model state, optimizer state, and scores
        score_list += [score_dict]
        score_df = pd.DataFrame(score_list)
        print('\n', score_df.tail(), '\n')
        if e % 10 == 0:
            hu.torch_save(os.path.join(savedir, 'model_state_dict.pth'),
                          model.get_state_dict())
        hu.save_pkl(os.path.join(savedir, 'score_list.pkl'), score_list)
        hu.save_pkl(os.path.join(savedir, 'meta_dict.pkl'), meta_dict)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)

    args = parser.parse_args()

    # Collect experiments
    if args.exp_id is not None:
        # Select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))
        
        exp_list = [exp_dict]
        
    else:
        # Select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Launch jobs on compute cluster
    if False:
        from haven import haven_jobs as hj
        run_command = ('python train.py -ei <exp_id> '
                       '-fid %d -sb %s -u %s -t %s' % 
                        (args.compute_fid, args.savedir_base, args.username,
                         args.use_tensorboard))
        hj.run_exp_list_jobs(
            exp_list,
            savedir_base=args.savedir_base, 
            workdir=os.path.dirname(os.path.realpath(__file__)),
            run_command=run_command,
            job_utils_path=exp_configs.JOB_UTILS_PATH,
            job_config=exp_configs.BORGY_CONFIGS[args.username])
    # Launch jobs locally
    else:
        # Run experiments
        for exp_dict in exp_list:
            train(exp_dict=exp_dict,
                  savedir_base=args.savedir_base,
                  reset=args.reset)

