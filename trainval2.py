import tqdm, job_config
import os

from haven import haven_examples as he
from haven import haven_wizard as hw
from haven import haven_results as hr


# 1. define the training and validation function
def trainval(exp_dict, savedir, args):
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

    # 3. load checkpoint
    chk_dict = hw.get_checkpoint(savedir)

    # 4. Add main loop
    for epoch in tqdm.tqdm(range(chk_dict['epoch'], 10),
                           desc="Running Experiment"):
        # 5. train for one epoch
        train_dict = model.train_on_loader(train_loader, epoch=epoch)

        # 6. get and save metrics
        score_dict = {'epoch':epoch, 'acc': train_dict['train_acc'],
                      'loss':train_dict['train_loss']}
        chk_dict['score_list'] += [score_dict]

        images = model.vis_on_loader(train_loader)

    hw.save_checkpoint(savedir, score_list=chk_dict['score_list'], images=[images])
    print('Experiment done\n')

# 7. create main
if __name__ == '__main__':
    # 8. define a list of experiments
    exp_list = []
    for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, "bug"]:
        exp_list += [{'lr':lr, 'dataset':'mnist', 'model':'linear'}]

    # 9. Launch experiments using magic command
    hw.run_wizard(func=trainval, exp_list=exp_list, job_config=job_config.JOB_CONFIG)
