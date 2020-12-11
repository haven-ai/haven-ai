from haven import haven_utils as hu

EXP_GROUPS = {}
EXP_GROUPS['group0'] = [{"dataset": 'mnist',
                        "batch_size": 128,
                        "opt": {'name': 'adam', 'lr': 1e-3}},
                        
                        {"dataset": 'fashionmnist',
                        "model": 'mlp',
                        "batch_size": 128,
                        "opt": {'name': 'adam', 'lr': 1e-3}}]

EXP_GROUPS['group1'] = hu.cartesian_exp_group({"dataset": ['mnist','fashionmnist'],
                        "batch_size": [128],
                        "opt": [
                                {'name': 'adam', 'lr': 1e-3},
                                {'name': 'sgd', 'lr': 1e-3},
                                {'name': 'adagrad', 'lr': 1e-3}
                                ],
        })

