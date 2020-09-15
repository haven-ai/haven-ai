from haven import haven_utils as hu


EXP_GROUPS = {}

EXP_GROUPS['mnist_full'] = hu.cartesian_exp_group({
        'batch_size': 32,
        'batch_size_val': 1024,
        'dataset': 'mnist_full',
        'max_epoch': 50,
        'max_cycle': 100,
        'opt':{'name':'sgd', 'lr':1e-3},
        'model': {'name':'clf', 'base':'lenet'},

        'active_learning': 
            [ {'ndata_to_label': 32,
             'name':'random',
             'ndata_to_label_init':32},

        {'ndata_to_label': 32,
                    'batch_size_pool':128,
                    'n_mcmc':50,
                    'name':'bald',
                    'ndata_to_label_init':32},


                    {'ndata_to_label': 32,
                    'batch_size_pool':128,
                    'n_mcmc':50,
                    'name':'entropy',
                    'ndata_to_label_init':32},             
             ]
        })

EXP_GROUPS['mnist_binary'] = hu.cartesian_exp_group({
        'batch_size': 32,
        'batch_size_val': 1024,
        'dataset': 'mnist_binary',
        'max_epoch': 50,
        'max_cycle': 100,
        'opt':{'name':'sgd', 'lr':1e-3},
        'model': {'name':'clf', 'base':'lenet'},

        'active_learning': 
            [ {'ndata_to_label': 32,
             'name':'random',
             'ndata_to_label_init':32},

        {'ndata_to_label': 32,
                    'batch_size_pool':128,
                    'n_mcmc':50,
                    'name':'bald',
                    'ndata_to_label_init':32},


                    {'ndata_to_label': 32,
                    'batch_size_pool':128,
                    'n_mcmc':50,
                    'name':'entropy',
                    'ndata_to_label_init':32},             
             ]
        })
