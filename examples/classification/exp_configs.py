from haven import haven_utils as hu

EXP_GROUPS = {'mnist':

                hu.cartesian_exp_group({
                    'lr':[1e-3, 1e-4],
                    'batch_size':[32, 64]})
                }