import numpy as np
import torch


def get_optim(opt_dict, parameters):
    opt_name = opt_dict['name']
    if opt_name == "adam":
        opt = torch.optim.Adam(parameters, lr=opt_dict['lr'],  betas=(0.9,0.99))

    elif opt_name == "adagrad":
        opt = torch.optim.Adagrad(parameters, lr=opt_dict['lr'])

    elif opt_name == 'sgd':
        opt = torch.optim.SGD(parameters, lr=opt_dict['lr'])

    elif opt_name == 'sps':
        opt = sps.Sps(parameters, c=.2, 
                        n_batches_per_epoch=n_batches_per_epoch, 
                        adapt_flag=opt_dict.get('adapt_flag', 'basic'),
                        fstar_flag=opt_dict.get('fstar_flag'),
                        eta_max=opt_dict.get('eta_max'),
                        eps=opt_dict.get('eps', 0))
    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt

