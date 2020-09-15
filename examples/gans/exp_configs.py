from haven import haven_utils as hu


EXP_GROUPS = {}

EXP_GROUPS['wgan'] = hu.cartesian_exp_group({
      'dataset': ['celeba', 'cifar10'],
      'base': ['wgan_resnet'],
      # can be one of['image', 'label', None]
      'conditioning_input': [None],
      # can be one of ['product', 'concat', 'condin', 'condbn', None]
      'conditioning_type': [None],
      # can be one of ['bn', 'ln', 'in ,'condin', 'condbn', None]
      'gen_norm': ['bn'],
      'disc_norm': [None],
      'gpu': ['0'],
      'batch_size': [64],
      'learning_rate': [0.0002],
      'lambda_gp': [10],
      'beta1': [0.5],
      'beta2': [0.999],
      'nz': [128],
      'ngf': [128],
      'ndf': [128],
      'nef': [128],
      'd_iterations': [5],
      'num_episodes': [1],
      'num_epochs': [200],
      'image_size': [32],
      'dataloader': ['standard'],
      'model': ['wgan']
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
