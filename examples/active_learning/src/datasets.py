#%%
# from haven import utils as mlkit_ut
import torchvision, torch
import numpy as np
from torchvision.transforms import transforms
from sklearn.utils import shuffle
from baal.utils.transforms import PILToLongTensor
from PIL import Image
import os
import os 
import numpy as np 

import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
from PIL import Image
import numpy as np
import torch
import os
from skimage.io import imread
from scipy.io import loadmat
import torchvision.transforms.functional as FT
import numpy as np
import torch
from skimage.io import imread
import torchvision.transforms.functional as FT
from skimage.transform import rescale
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import pylab as plt
from skimage.color import label2rgb
from torch import nn


def get_dataset(dataset_name, split, datadir_base='', exp_dict=None):
    # load dataset
    if dataset_name == "mnist_binary":
        dataset = Mnist(split=split, binary=True, datadir_base=datadir_base)

    if dataset_name == "mnist_full":
        dataset = Mnist(split=split, binary=False, datadir_base=datadir_base)
        
    return dataset

class Mnist:
    def __init__(self, split, binary=False, datadir_base=None):
        self.split = split
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5,), (0.5,))
        ])

        if split == "train":
            dataset = torchvision.datasets.MNIST(datadir_base, train=True,
                                               download=True,
                                               transform=transform)
        elif split == "val":
            dataset = torchvision.datasets.MNIST(datadir_base, train=False,
                                               download=True,
                                               transform=transform)
        # get only two classes
        if binary:
            ind_class2 = dataset.targets == 2
            ind_class8 = dataset.targets == 8

            dc = torch.cat([dataset.data[ind_class2], dataset.data[ind_class8]])
            tc = torch.cat([dataset.targets[ind_class2], dataset.targets[ind_class8]])

            ind_shuffle = torch.randperm(dc.shape[0])
            dataset.data = dc[ind_shuffle]
            dataset.targets = tc[ind_shuffle]

        self.dataset = dataset 
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        images, labels = self.dataset[index]
        
        batch = {"images": images,
                 "labels": labels,
                 "meta": {"index": index,
                          "image_id": index,
                          "split": self.split}}
        return batch


