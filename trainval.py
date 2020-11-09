import tqdm, os

from haven import haven_examples as he
from haven import haven_wizard as hw

# 3. define trainval function
def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # 4. Create data loader and model 
    train_loader = he.get_loader(name=exp_dict['dataset'], split='train', 
                                 datadir=os.path.dirname(savedir),
                                 exp_dict=exp_dict)
    model = he.get_model(name=exp_dict['model'], exp_dict=exp_dict)

    # 5. load checkpoint
    chk_dict = hw.get_checkpoint(savedir)

    # 6. Add main loop
    for epoch in range(chk_dict['epoch'], 3):
        # 7. train for one epoch
        for batch in tqdm.tqdm(train_loader):
            train_dict = model.train_on_batch(batch)

        # 8. get and save metrics
        score_dict = {'epoch':epoch, 'acc': train_dict['train_acc'], 'loss':train_dict['train_loss']}
        chk_dict['score_list'] += [score_dict]
        hw.save_checkpoint(savedir, score_list=chk_dict['score_list'])

    print('Experiment done')

# 0. create main
if __name__ == '__main__':
    # 1. define a list of experiments
    exp_list = [{'dataset':'mnist', 'model':'linear', 'lr':lr} 
                for lr in [1e-3, 1e-4]]

    # 2. Launch experiments using magic command
    hw.run_wizard(func=trainval, exp_list=exp_list, results_fname='trainval_results.ipynb')