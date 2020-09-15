
import torch
from . import wgan
from src.models import base
import tqdm
import pandas as pd
import src.utils as ut
import os
from haven import haven_utils as hu


def get_model(model_name, num_train_classes, num_eval_classes, 
              num_channels, device, exp_dict):    
    if model_name == 'wgan':
        netG, netD = base.get_base(exp_dict['base'], num_channels,
                               num_train_classes, num_eval_classes,
                               exp_dict)
        netD = netD.to(device)
        netG = netG.to(device)
        optD = torch.optim.Adam(netD.parameters(), lr=exp_dict['learning_rate'],
                                betas=(exp_dict['beta1'], exp_dict['beta2']))
        optG = torch.optim.Adam(netG.parameters(), lr=exp_dict['learning_rate'],
                                betas=(exp_dict['beta1'], exp_dict['beta2']))
        model = wgan.WGan(netG, netD, optG, optD, device,
                            image_size=exp_dict['image_size'],
                            batch_size=exp_dict['batch_size'],
                            lambda_gp=exp_dict['lambda_gp'],
                            d_iterations=exp_dict['d_iterations'])

    return model



def standard_eval_on_loader(model, eval_loader, save_dir, epoch):
    print('Evaluating...')
    if eval_loader is not None:
        # Grab first batch from loader, ignore the rest
        batch = iter(eval_loader).next()
        model.eval_on_batch(batch, save_dir, epoch)
    else:
        model.eval_on_batch(None, save_dir, epoch)
    return {}
