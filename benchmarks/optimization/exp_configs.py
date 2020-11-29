from haven import haven_utils as hu
RUNS = [0, 1, 2]
EXP_GROUPS = {}

adam_constant_list = []
adam_constant_list += [
    {'name': 'adam', 'lr': 1e-3, 'betas': [0, 0.99]}
]

amsgrad_constant_list = []
amsgrad_constant_list += [
    {'name': 'adam', 'lr': 1e-3, 'betas': [0, 0.99], 'amsgrad':True}
]

sls_list = [{'name': "sgd_armijo", 'c': .2, 'reset_option': 1}]

sgd = [{'name': 'sgd', 'lr': 1e-3}]

adagrad = [{'name': 'adagrad', 'lr': 1e-3}]


# sps_list = []
# c_list = [.2, .5, 1.0]
# for c in c_list:
#     sps_list += [{'name': "sps", 'c': c}]

opt_list = [] + sgd + adam_constant_list + sls_list + amsgrad_constant_list + adagrad

EXP_GROUPS['mnist'] = {"dataset": {'name': 'mnist'},
        "model": {'name': 'mlp'},
        "runs": RUNS,
        "batch_size": [128],
        "max_epoch": [100],
        'dataset_size': [
            {'train': 'all', 'val': 'all'},
        ],
        "loss_func": ["softmax_loss"],
        "opt": opt_list,
        "acc_func": ["softmax_accuracy"],
        }

EXP_GROUPS['syn'] = {"dataset": {'name': 'synthetic'},
        "model": {'name': 'logistic'},
        "runs": RUNS,
        "batch_size": [128],
        "max_epoch": [200],
        'dataset_size': [
            {'train': 'all', 'val': 'all'},
        ],
        "loss_func": ["softmax_loss"],
        "opt": opt_list,
        "acc_func": ["softmax_accuracy"],
        'margin': [0.1],
        "n_samples": [1000],
        "d": 20
        }

EXP_GROUPS['cifar10'] = {"dataset": {'name': 'cifar10'},
        "model": {'name': "densenet121"},
        "runs": RUNS,
        "batch_size": [128],
        "max_epoch": [200],
        'dataset_size': [
            {'train': 'all', 'val': 'all'},
        ],
        "loss_func": ["softmax_loss"],
        "opt": opt_list,
        "acc_func": ["softmax_accuracy"]
        }

# TODO: may need more info for each optimizer
EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}
