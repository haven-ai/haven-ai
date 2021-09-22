import torch
import torchvision
import os
import pprint
import tqdm
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from haven import haven_utils as hu
from haven import haven_wizard as hw
from torchvision import datasets, transforms, models


def get_model(name, exp_dict):
    if name == "linear":
        model = Linear()
        model.opt = torch.optim.Adam(model.parameters(), lr=exp_dict["lr"])

    if name == "mlp":
        model = Mlp()
        model.opt = torch.optim.Adam(model.parameters(), lr=exp_dict["lr"])

    if name == "seg_model":
        model = Mlp()
        model.opt = torch.optim.Adam(model.parameters(), lr=exp_dict["lr"])

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
        logits = self.model.forward(images.view(images.shape[0], -1).cuda())
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(logits, labels.view(-1).cuda())
        loss.backward()

        self.opt.step()

        return {
            "train_loss": loss.item(),
            "train_acc": (logits.argmax(dim=1) == labels.cuda()).float().mean().item(),
        }

    @torch.no_grad()
    def vis_on_loader(self, loader, **extras):
        self.eval()

        for batch in loader:
            images, labels = batch
            probs = torch.softmax(self.model.forward(images.view(images.shape[0], -1)), dim=1)
            score, label = probs.max(dim=1)
            i_list = []
            for i in range(probs.shape[0]):
                pil_img = hu.save_image("tmp", images[i], return_image=True)
                img = get_image(pil_img, "Predicted %d (Prob: %.2f)" % (label[i], score[i]))
                i_list += [img]
                if i > 5:
                    break
            return np.hstack(i_list)[:, :, None].repeat(3, axis=2)

    def train_on_loader(self, loader, **extras):
        for batch in tqdm.tqdm(loader, desc="Epoch %d" % extras.get("epoch"), leave=False):
            train_dict = self.train_on_batch(batch)

        return train_dict


# MLP


class Mlp(torch.nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_sizes=[512, 256],
        n_classes=10,
        bias=True,
        dropout=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_size, out_size, bias=bias)
                for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            ]
        )
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

        return {
            "train_loss": loss.item(),
            "train_acc": (logits.argmax(dim=1) == labels).float().mean().item(),
        }

    def train_on_loader(self, loader, **extras):
        for batch in tqdm.tqdm(loader, desc="Epoch %d" % extras.get("epoch"), leave=False):
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
        state_dict = {"model": self.state_dict(), "opt": self.opt.state_dict()}

        return state_dict

    def set_state_dict(self, state_dict):
        self.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])
