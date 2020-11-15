import torch, torchvision, os, pprint
import tqdm
import argparse, pandas as pd

from torch.utils.data import TensorDataset
from haven import haven_utils as hu
from haven import haven_wizard as hw

def get_loader(name, split, datadir, exp_dict):
    if name == 'syn':
        # get dataset and loader
        X = torch.randn(5000, 1, 28, 28)
        y = torch.randint(0, 2, (5000,1))
        dataset = TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    if name == 'mnist':
        # get dataset and loader
        train = True if split == 'train' else False
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        dataset = torchvision.datasets.MNIST(root=datadir, 
                                             train=train, download=True,
                                             transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256)

    return loader

def get_model(name, exp_dict):
    if name == 'linear':
        model = Linear()
        model.opt = torch.optim.Adam(model.parameters(), lr=exp_dict['lr'])

    if name == 'mlp':
        model = Mlp()
        model.opt = torch.optim.Adam(model.parameters(), lr=exp_dict['lr'])

    return model

# =====================================================
class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(784, 10)

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()

        images, labels = batch
        logits = self.model.forward(images.view(images.shape[0], -1))
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(logits, labels.view(-1))
        loss.backward()

        self.opt.step()

        return {"train_loss": loss.item(), 'train_acc':(logits.argmax(dim=1) == labels).float().mean().item()}
    
    def train_on_loader(self, loader, **extras):
        for batch in tqdm.tqdm(loader, desc="Epoch %d" % extras.get('epoch'), 
                               leave=False):
            train_dict = self.train_on_batch(batch)

        return train_dict

# MLP
class Mlp(torch.nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[512, 256],
                 n_classes=10,
                 bias=True, dropout=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(in_size, out_size, bias=bias) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], n_classes, bias=bias)

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()

        images, labels = batch
        logits = self.forward(images)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(logits, labels.view(-1))
        loss.backward()

        self.opt.step()

        return {"train_loss": loss.item(), 'train_acc':(logits.argmax(dim=1) == labels).float().mean().item()}

    def train_on_loader(self, loader, **extras):
        for batch in tqdm.tqdm(loader, desc="Epoch %d" % extras.get('epoch'), 
                               leave=False):
            train_dict = self.train_on_batch(batch)

        return train_dict

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = torch.nn.functional.relu(Z)

        logits = self.output_layer(out)

        return logits

    def get_state_dict(self):
        state_dict = {"model": self.state_dict(),
                      "opt":self.opt.state_dict()}

        return state_dict
        
    def set_state_dict(self, state_dict):
        self.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])




    