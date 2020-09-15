# Compare between two learning rates for the same model and dataset
EXP_GROUPS = {'mnist':
                [
                 {'lr':1e-3, 'model':'mlp', 'dataset':'mnist'},
                 {'lr':1e-4, 'model':'mlp', 'dataset':'mnist'}
                  ]
}