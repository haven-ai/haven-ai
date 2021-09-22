import torch
import torchvision
import os
from PIL import Image
import pprint
import tqdm, glob
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from haven import haven_utils as hu
from haven import haven_wizard as hw


def get_dataset(name, split, datadir, exp_dict, download=True):
    if name == "syn":
        # get dataset and loader
        X = torch.randn(5000, 1, 28, 28)
        y = torch.randint(0, 2, (5000, 1))
        dataset = TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    if name == "mnist":
        # get dataset and loader
        train = True if split == "train" else False
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(root=datadir, train=train, download=download, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    if name == "pascal_small":
        # get dataset and loader
        train = True if split == "train" else False
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(root=datadir, train=train, download=download, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    return dataset
