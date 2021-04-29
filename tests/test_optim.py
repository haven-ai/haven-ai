# Import Libraries
import sys, os

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import pandas as pd
import torch, copy, pprint
import numpy as np
import os, shutil, torchvision
import tqdm.notebook as tqdm
import sklearn
import torch.nn.functional as F

from sklearn import preprocessing
from sklearn.datasets import load_iris
from haven import haven_results as hr
from haven import haven_utils as hu
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# from torchnlp.datasets.imdb import imdb_dataset

# Define a set of datasets
# ------------------------
def get_dataset(dataset_dict, split):
    name = dataset_dict["name"]
    if name == "syn":
        X, y = hu.make_binary_linear(n=1000, d=49, margin=0.5, separable=dataset_dict["separable"])
        dataset = hu.get_split_torch_dataset(X, y, split)
        dataset.task = "binary_classification"
        dataset.n_output = 1

        return dataset

    if name == "iris":
        # X, y = sklearn.datasets.load_iris(return_X_y=True)
        X, y = load_iris(return_X_y=True)
        X = preprocessing.StandardScaler().fit_transform(X)
        dataset = hu.get_split_torch_dataset(X, y, split)
        dataset.task = "multi_classification"
        dataset.n_output = 3

        return dataset

    if name == "diabetes":
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        y = y / y.max()
        X = preprocessing.StandardScaler().fit_transform(X)
        dataset = hu.get_split_torch_dataset(X, y.astype("float"), split)
        dataset.task = "regression"
        dataset.n_output = 1
        return dataset

    if name == "imdb":
        train = True if split == "train" else False
        test = True if split == "val" else False
        dataset = imdb_dataset(train=train, test=test)
        X = [d["text"] for d in dataset]
        y = [d["sentiment"] for d in dataset]

        dataset.task = "classification"
        dataset.n_output = 1
        return dataset

    if name == "fashion_mnist":
        train = True if split == "train" else False
        dataset = torchvision.datasets.FashionMNIST(
            "data/",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        dataset.n_input = 784
        dataset.task = "multi_classification"
        dataset.n_output = 10
        return dataset

    if name == "mnist":
        train = True if split == "train" else False
        dataset = torchvision.datasets.MNIST(
            "data/",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        dataset.n_input = 784
        dataset.task = "multi_classification"
        dataset.n_output = 10
        return dataset


# Define a set of optimizers
# --------------------------
def get_optimizer(opt_dict, model, train_set, batch_size):
    name = opt_dict["name"]
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=opt_dict.get("lr", 1e-3))

    elif name == "adasls":
        return adasls.AdaSLS(
            model.parameters(),
            c=opt_dict.get("c", 0.5),
            n_batches_per_epoch=opt_dict.get("n_batches_per_epoch", len(train_set) / batch_size),
        )
    elif name == "sls":
        return sls.Sls(
            model.parameters(),
            c=opt_dict.get("c", 0.5),
            n_batches_per_epoch=opt_dict.get("n_batches_per_epoch", len(train_set) / batch_size),
        )
    elif name == "sps":
        return sps.Sps(model.parameters(), c=opt_dict.get("c", 0.5))

    elif name == "sgd":
        return torch.optim.Sgd(model.parameters(), lr=opt_dict.get("lr", 1e-3), momentum=opt_dict.get("momentum", 0.0))

    elif name == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), lr=opt_dict.get("lr", 1), line_search_fn="strong_wolfe")


# Define a set of models
# ----------------------
def get_model(model_dict, dataset):
    name = model_dict["name"]
    if name == "mlp":
        return MLP(dataset, model_dict["layer_list"])


class MLP(nn.Module):
    def __init__(self, dataset, layer_list):
        super().__init__()
        self.task = dataset.task
        layer_list = [dataset.n_input] + layer_list + [dataset.n_output]

        layers = [nn.Flatten()]
        for i in range(len(layer_list) - 1):
            layers += [nn.Linear(layer_list[i], layer_list[i + 1])]

        self.layers = nn.Sequential(*layers)
        self.n_forwards = 0

    def forward(self, x):
        return self.layers(x)

    def compute_loss(self, X, y):
        # Compute the loss based on the task
        logits = self(X)

        if self.task == "binary_classification":
            func = nn.BCELoss()
            loss = func(logits.sigmoid().view(-1), y.float().view(-1))
        if self.task == "multi_classification":
            func = nn.CrossEntropyLoss()
            loss = func(logits.softmax(dim=1), y)

        if self.task == "regression":
            func = nn.MSELoss()
            loss = F.mse_loss(logits.view(-1), y.float().view(-1))

        # Add L2 loss
        w = 0.0
        for p in self.parameters():
            w += (p ** 2).sum()
        loss += 1e-4 * w

        return loss

    def compute_score(self, X, y):
        # Computes the score based on the task
        logits = self(X)

        if self.task == "binary_classification":
            y_hat = (logits.sigmoid().view(-1) > 0.5).long()
            return (y_hat == y.view(-1)).sum()

        if self.task == "multi_classification":
            y_hat = logits.softmax(dim=1).argmax(dim=1).long()
            return (y_hat == y).sum()

        if self.task == "regression":
            return F.mse_loss(logits.view(-1), y.float().view(-1))

    def compute_metrics(self, dataset):
        metric_list = []
        n = len(dataset)
        loader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False)

        for batch in loader:
            # get batch
            Xi, yi = batch
            # compute loss & acc
            loss = self.compute_loss(Xi, yi)
            score = self.compute_score(Xi, yi)
            # aggregate scores
            metric_list += [{"loss": float(loss) / n, "score": float(score) / n}]
        metric_dict = pd.DataFrame(metric_list).sum().to_dict()

        return metric_dict


def trainval(exp_dict):
    # set seed
    seed = 5 + exp_dict["run"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # print exp dict
    savedir = f"{savedir_base}/{hu.hash_dict(exp_dict)}"
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)

    # Get datasets
    train_set = get_dataset(exp_dict["dataset"], split="train")
    val_set = get_dataset(exp_dict["dataset"], split="val")

    # sample n_max examples
    n_max = exp_dict["dataset"]["n_max"]

    if n_max == -1 or n_max >= len(train_set):
        ind_list = np.arange(len(train_set))
        n_max = len(train_set)
    else:
        ind_list = np.random.choice(len(train_set), n_max, replace=False)

    train_set = torch.utils.data.Subset(train_set, ind_list)

    # choose full or mini-batch
    batch_size = exp_dict["opt"]["batch_size"]
    if batch_size < 0:
        batch_size = n_max
    batch_size = min(batch_size, len(train_set))

    print(
        f'Dataset: {exp_dict["dataset"]["name"]} ({len(train_set)}) '
        f'- Model: {exp_dict["model"]["name"]} - '
        f'Opt: {exp_dict["opt"]["name"]} ({batch_size})'
    )

    # get loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    # Load model and optimizer
    model = get_model(exp_dict["model"], train_set.dataset)
    opt = get_optimizer(exp_dict["opt"], model, train_set.dataset, batch_size)

    score_list = []
    # start training and validating
    ebar = tqdm.tqdm(range(exp_dict["epochs"]), leave=False)

    model.n_calls = 0.0
    for e in ebar:
        # Compute Metrics on Validation and Training Set
        val_dict = model.compute_metrics(val_set)
        train_dict = model.compute_metrics(train_set)

        # Train a single epoch

        for batch in train_loader:
            # get batch
            Xi, yi = batch

            # define closure
            def closure():
                loss = model.compute_loss(Xi, yi)
                if exp_dict["opt"]["name"] not in ["adasls", "sls"]:
                    loss.backward()
                model.n_calls += Xi.shape[0]
                # print(Xi.shape[0])

                return loss

            # update parameters
            opt.zero_grad()
            loss = opt.step(closure=closure)

        # Update and save metrics
        score_dict = {}
        score_dict["epoch"] = e
        score_dict["val_score"] = val_dict["score"]
        score_dict["val_loss"] = val_dict["loss"]
        score_dict["train_loss"] = train_dict["loss"]
        score_dict["n_train"] = len(train_set)
        score_dict["step_size"] = opt.state.get("step_size", {})

        n_iters = len(train_loader) * (e + 1)
        score_dict["n_calls"] = int(model.n_calls)
        score_dict["n_backwards"] = opt.state.get("n_backwards", n_iters)
        score_list += [score_dict]

        # Save metrics
        hu.save_pkl(os.path.join(savedir, "score_list.pkl"), score_list)
        ebar.update(1)
        ebar.set_description(f'Training Loss {train_dict["loss"]:.3f}')


if __name__ == "__main__":
    # Specify the hyperparameters
    # dataset_list = [{'name':'syn', 'separable':True, 'n_max':500}]
    run_list = [0, 1]
    dataset_list = [{"name": "diabetes", "n_max": -1}]
    dataset_list = [{"name": "iris", "n_max": -1}]
    # dataset_list = [{'name':'mnist', 'n_max':1000}]
    model_list = [{"name": "mlp", "layer_list": []}]
    opt_list = [
        # {'name':'adasls', 'c':.5,  'batch_size':128},
        {"name": "lbfgs", "lr": 1, "batch_size": -1},
        # {'name':'adam', 'lr':1e-3, 'batch_size':128},
        # {'name':'adam', 'lr':1e-4, 'batch_size':128},
        # {'name':'sps', 'c':.5, 'batch_size':128},
        # {'name':'sls', 'c':.5, 'batch_size':128}
    ]

    # Create experiments
    exp_list = []
    for dataset in dataset_list:
        for model in model_list:
            for opt in opt_list:
                for run in run_list:
                    exp_list += [{"dataset": dataset, "model": model, "opt": opt, "epochs": 20, "run": run}]
    print(f"Defined {len(exp_list)} experiments")

    # Create main save directory
    savedir_base = ".tmp/results"
    if os.path.exists(savedir_base):
        shutil.rmtree(savedir_base)

    # Run each experiment and save their results
    pbar = tqdm.tqdm(exp_list)
    for ei, exp_dict in enumerate(pbar):
        pbar.set_description(f"Running Exp {ei+1}/{len(exp_list)} ")
        trainval(exp_dict)

        # Update progress bar
        pbar.update(1)

    # Plot results
    rm = hr.ResultManager(exp_list=exp_list, savedir_base=savedir_base, verbose=0)
    # rm.get_plot_all(y_metric_list=['train_loss', 'val_loss', 'val_score', 'step_size'],
    #             x_metric='epoch', figsize=(18,4), title_list=['dataset.name'],
    #             legend_list=['opt'], groupby_list=['dataset'],
    #             log_metric_list=['train_loss', 'val_loss'], avg_across='run')

    rm.get_plot_all(
        y_metric_list=["train_loss", "val_loss", "val_score", "step_size"],
        x_metric="n_calls",
        figsize=(18, 4),
        title_list=["dataset.name"],
        legend_list=["opt"],
        groupby_list=["dataset"],
        log_metric_list=["train_loss", "val_loss"],
        avg_across="run",
    )
