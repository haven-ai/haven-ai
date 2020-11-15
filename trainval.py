import tqdm, os

from haven import haven_examples as he
from haven import haven_wizard as hw

# 0. define trainval function
def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # 1. Create data loader and model 
    train_loader = he.get_loader(name=exp_dict['dataset'], split='train', 
                                 datadir=os.path.dirname(savedir),
                                 exp_dict=exp_dict)
    model = he.get_model(name=exp_dict['model'], exp_dict=exp_dict)

    # 2. load checkpoint
    chk_dict = hw.get_checkpoint(savedir)

    # 3. Add main loop
    for epoch in tqdm.tqdm(range(chk_dict['epoch'], 3), desc="Running experiment"):
        # 4. train for one epoch
        train_dict = model.train_on_loader(train_loader, epoch=epoch)

        # 5. get and save metrics
        score_dict = {'epoch':epoch, 'acc': train_dict['train_acc'], 
                        'loss':train_dict['train_loss']}
        chk_dict['score_list'] += [score_dict]
        hw.save_checkpoint(savedir, score_list=chk_dict['score_list'])

    from haven import haven_results as hr
    rm = hr.ResultManager(exp_list=None, 
                      savedir_base=os.path.split(savedir)[0], 
                      filterby_list=None,
                      verbose=0,
                      exp_groups=None
                     )
    print('Experiment done')

# 6. create main
if __name__ == '__main__':
    # 7. define a list of experiments
    exp_list = [{'dataset':'syn', 'model':'linear', 'lr':lr, 'yours':{'name':'hello', 'lb':3}} 
                for lr in [1e-3, 1e-4]]

    # 8. Launch experiments using magic command
    hw.run_wizard(func=trainval, 
                  exp_list=exp_list, 
                  job_config={
                            'account_id': os.environ['EAI_ACCOUNT_ID'],
                            'image': 'registry.console.elementai.com/eai.colab/ssh',
                            'data': ['eai.colab.public:/mnt/public' ],
                            'restartable':True,
                            'resources': {
                                'cpu': 4,
                                'mem': 8,
                                'gpu': 1
                            },
                            'interactive': False,
                            'bid':9999,
                            })