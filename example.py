import torch, torchvision, os, pprint
import tqdm
import argparse, pandas as pd

from haven import haven_utils as hu
from haven import haven_wizard as hw

def trainval(exp_dict, savedir):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    """
    # get dataset and loader
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root=os.path.dirname(savedir), train=False, download=True,
                                         transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    # get model and optimizer
    model = torch.nn.Linear(784, 10, bias=False)
    opt = torch.optim.Adam(model.parameters(), lr=exp_dict['lr'])

    # run training loop
    chk_dict = hw.get_checkpoint(savedir, return_model_state_dict=True)

    if len(chk_dict['model_state_dict']):
        model.load_state_dict(chk_dict['model_state_dict']['model'])
        opt.load_state_dict(chk_dict['model_state_dict']['opt'])

    for e in range(chk_dict['epoch'], exp_dict['max_epoch']):
        for batch in tqdm.tqdm(train_loader):
            images, labels = batch

            # train on batch
            logits = model.forward(images.view(images.shape[0], -1))
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

            # update optimizer
            opt.zero_grad()
            loss.backward()
            opt.step()

        # get score dict
        score_dict = {'loss': float(loss), 'epoch':e}

        # save score_list and model
        chk_dict['score_list'] += [score_dict]
        hw.save_checkpoint(savedir, score_list=chk_dict['score_list'], 
                                    model_state_dict={'model':model.state_dict(),
                                                      'opt':opt.state_dict()})

    print('Experiment done')

JOB_CONFIG = {'image': 'registry.console.elementai.com/%s/ssh' % 
                      os.environ['EAI_ACCOUNT_ID'] ,
      'data': ['eai.colab.public:/mnt/public'],
      'restartable':True,
      'resources': {
          'cpu': 4,
          'mem': 8,
          'gpu': 1
      },
      'interactive': False,
      'bid':9999,
      }

if __name__ == '__main__':
    # define a list of experiments
    exp_list = []
    for lr in [1e-1, 1e-4, 1e-10]:
        exp_dict = {'dataset': 'mnist', 
                    'model': 'logistic', 
                    'max_epoch': 3,
                    'lr':lr}

        exp_list += [exp_dict]

    # run experiments
    hw.run_wizard(func=trainval, exp_list=exp_list)
