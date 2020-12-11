from haven import haven_utils as hu

RUNS = [0, 1, 2]

opt_list = [
     {'name': 'adam', 'lr': 1e-3},
     {'name': 'sgd', 'lr': 1e-3},
     {'name': 'adagrad', 'lr': 1e-3}
    ]

EXP_GROUPS = {}

EXP_GROUPS['mnist'] = {"dataset": 'mnist',
                        "model": 'mlp',
                        "runs": RUNS,
                        "batch_size": [128],
                        "opt": opt_list,
        }

EXP_GROUPS['fashionmnist'] = {"dataset": 'fashionmnist',
                        "model": 'mlp',
                        "runs": RUNS,
                        "batch_size": [128],
                        "opt": opt_list,
        }


EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}
