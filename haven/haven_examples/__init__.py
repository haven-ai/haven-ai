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


def get_loader(name, split, datadir, exp_dict, download=True):
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

    return loader


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
        logits = self.model.forward(images.view(images.shape[0], -1))
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(logits, labels.view(-1))
        loss.backward()

        self.opt.step()

        return {"train_loss": loss.item(), "train_acc": (logits.argmax(dim=1) == labels).float().mean().item()}

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
    def __init__(self, input_size=784, hidden_sizes=[512, 256], n_classes=10, bias=True, dropout=False):
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

        return {"train_loss": loss.item(), "train_acc": (logits.argmax(dim=1) == labels).float().mean().item()}

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


def get_image(pil_img, title):
    import pylab as plt
    from PIL import Image

    # plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("tmp.png", bbox_inches="tight", pad_inches=0)
    plt.close()
    return np.array(Image.open("tmp.png").convert("L"))


def save_example_results(savedir_base="results"):
    import os
    import pandas
    import requests
    import io
    import matplotlib.pyplot as plt

    from .. import haven_results as hr
    from .. import haven_utils as hu
    from PIL import Image

    # create hyperparameters
    exp_list = [{"dataset": "mnist", "model": "mlp", "lr": lr} for lr in [1e-1, 1e-2, 1e-3]]

    for i, exp_dict in enumerate(exp_list):
        # get hash for experiment
        exp_id = hu.hash_dict(exp_dict)

        # add scores for loss, and accuracy
        score_list = []
        for e in range(1, 10):
            score_list += [{"epoch": e, "loss": 1 - e * exp_dict["lr"] * 0.9, "acc": e * exp_dict["lr"] * 0.1}]
        # save scores and images
        hu.save_json(os.path.join(savedir_base, exp_id, "exp_dict.json"), exp_dict)
        hu.save_pkl(os.path.join(savedir_base, exp_id, "score_list.pkl"), score_list)

        url = "https://raw.githubusercontent.com/haven-ai/haven-ai/master/haven/haven_examples/data/%d.png" % (i + 1)
        response = requests.get(url).content
        img = plt.imread(io.BytesIO(response), format="JPG")
        hu.save_image(os.path.join(savedir_base, exp_id, "images/1.png"), img[:, :, :3])
