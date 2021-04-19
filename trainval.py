import tqdm, argparse
import os

from haven import haven_examples as he
from haven import haven_wizard as hw
from haven import haven_results as hr
from haven import haven_utils as hu


# 1. define the training and validation function
def trainval(exp_dict, savedir, args, logger=None):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # 2. Create data loader and model 
    train_loader = he.get_loader(name=exp_dict['dataset'], split='train', 
                                 datadir=os.path.dirname(savedir),
                                 exp_dict=exp_dict)
    model = he.get_model(name=exp_dict['model'], exp_dict=exp_dict)
    
    # 3. resume or initialize checkpoint
    chk_dict = hw.get_checkpoint(savedir)
    if len(chk_dict['model_state_dict']):
        model.set_state_dict(chk_dict['model_state_dict'])

    # 4. Add main loop
    for epoch in tqdm.tqdm(range(chk_dict['epoch'], 3), 
                           desc="Running Experiment"):
        # 5. train for one epoch
        train_dict = model.train_on_loader(train_loader, epoch=epoch)

        # 6. get and save metrics
        score_dict = {'epoch':epoch, 'acc': train_dict['train_acc'], 
                      'loss':train_dict['train_loss']}
        chk_dict['score_list'] += [score_dict]
        
        hu.wandb_logging(logger, score_dict)

        images = model.vis_on_loader(train_loader)

        hw.save_checkpoint(savedir, score_list=chk_dict['score_list'], images=[images])
    
    print('Experiment done\n')

# 7. create main
if __name__ == '__main__':
    # 8. define a list of experiments
    exp_list = []
    for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        exp_list += [{'lr':lr, 'dataset':'mnist', 'model':'linear'}]

    # 9. Launch experiments using magic command
    parser = argparse.ArgumentParser()
    
    group = parser.add_argument_group("General arguments")
    parser.add_argument('-sb', '--savedir_base', default=None,
                        help='Define the base directory where the experiments will be saved.')
    parser.add_argument("-r", "--reset",  default=0, type=int,
                        help='Reset or resume the experiment.')
    parser.add_argument("-j", "--job_scheduler",  default=None,
                        help='Choose Job Scheduler.')
    
    group = parser.add_argument_group("WandB arguments")
    parser.add_argument("-wb", "--wandb_activate", type=bool, default=False,
                        help="activate WandB monitoring")
    parser.add_argument("-wbp", "--wandb_project", type=str, default=None,
                        help="name of the WandB project to save your runs in")
    parser.add_argument("-wbn", "--wandb_name", type=str, default=None,
                        help="name of the run. can be used to group some runs together")
    parser.add_argument("-wbe", "--wandb_entity", type=str, default=None,
                        help="for submitting runs on shared WandB account")
    parser.add_argument("-wbk", "--wandb_key", type=str, default=None,
                        help="WandB auth key, if you can't use $(wandb login)")
    parser.add_argument("-wbkl", "--wandb_key_loc", type=str, default=None,
                        help="WandB auth key location (if you don't want to pass the key as an argument, store it in this location)")

    args, others = parser.parse_known_args()

    
    if args.job_scheduler == "slurm":
        import slurm_config
        job_config = slurm_config.JOB_CONFIG
    elif args.job_scheduler == "toolkit":
        import job_configs
        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    hw.run_wizard(func=trainval, exp_list=exp_list, savedir_base=args.savedir_base, reset=args.reset,
                  job_config=job_config)