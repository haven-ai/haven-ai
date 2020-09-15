
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import Subset
import src.utils as ut


def get_dataloader(loader_name, dataset_train, dataset_test, exp_dict):
    

    # ========================================================
    # Standard loaders
    # ========================================================
    if loader_name == 'standard':
        loader_train = \
            DataLoader(dataset_train, shuffle=True,
                       batch_size=exp_dict['batch_size'],
                       sampler=None,
                       pin_memory=True,
                       num_workers=0)

        loader_test = None

        return loader_train, loader_test
