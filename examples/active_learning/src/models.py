import torch
from torch import nn
import tqdm
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import numpy as np
import time
from src import active_learning as al
from sklearn.metrics import confusion_matrix
import skimage
from haven import haven_utils as hu
from torchvision import transforms
from src import models

def get_model(model_name, exp_dict):
    if model_name == "clf":
        if exp_dict['model']['base'] == 'lenet':
            base = LeNeT()
        model = ClfModel(base)

    return model

class ClfModel(torch.nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.model_base = model_base
        self.opt = torch.optim.SGD(self.parameters(), lr=1e-3)
    
    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt":self.opt.state_dict()}

        return state_dict

    def set_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])

    def train_on_loader(model, train_loader):
        model.train()

        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        train_monitor = TrainMonitor()
        for e in range(1):
            for i, batch in enumerate(train_loader):
                score_dict = model.train_on_batch(batch)
                
                train_monitor.add(score_dict)
                if i % 10 == 0:
                    msg = "%d/%d %s" % (i, n_batches, train_monitor.get_avg_score())
                    pbar.update(10)
                
                    pbar.set_description(msg)
        pbar.close()

        return train_monitor.get_avg_score()

    @torch.no_grad()
    def val_on_loader(model, val_loader, val_monitor):
        model.eval()

        n_batches = len(val_loader)
        
        pbar = tqdm.tqdm(desc="Validating", total=n_batches, leave=False)

        for i, batch in enumerate(val_loader):
            score = model.val_on_batch(batch)

            val_monitor.add(score)
            if i % 10 == 0:
                msg = "%d/%d %s" % (i, n_batches, val_monitor.get_avg_score())
                pbar.update(10)
                # print(msg)
                pbar.set_description(msg)
        pbar.close()

        return val_monitor.get_avg_score()

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        
        labels = batch["labels"].cuda()
        logits = self.model_base.forward(batch["images"].cuda())
        loss_clf =  F.cross_entropy(logits.squeeze(),
                        labels.squeeze(), reduction="mean")
        loss_clf.backward()

        self.opt.step()

        return {"train_loss":loss_clf.item()}

    def val_on_batch(self, batch, **extras):
        pred_clf = self.predict_on_batch(batch)
        return (pred_clf.cpu().numpy().ravel() != batch["labels"].numpy().ravel())
        
    def predict_on_batch(self, batch):
        images = batch["images"].cuda()
        n = images.shape[0]
        logits = self.model_base.forward(images)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def score_on_batch(self, batch, active_learning_dict):
        if active_learning_dict['name'] == 'entropy':
            probs_mcmc = self.mcmc_on_batch(batch, active_learning_dict)
            entropy = - al.xlogy(probs_mcmc).mean(dim=0).sum(dim=1)
            scores = entropy 

        elif active_learning_dict['name'] == 'bald':
            # mean over mcmc and sum over classes
            probs_mcmc = self.mcmc_on_batch(batch, active_learning_dict)
            entropy = al.xlogy(probs_mcmc).mean(dim=0).sum(dim=1)
            entropy_avg = al.xlogy(probs_mcmc.mean(dim=0)).sum(dim=1)

            scores = - (entropy + entropy_avg)

        else:
            raise

        return scores

    def get_active_indices(self, active_set, active_learning_dict, sampler=None):
        if active_learning_dict["name"] == "random":
            indices = np.random.choice(len(active_set.pool), 
                            active_learning_dict['ndata_to_label'])
            return indices

        else:
            pool_loader = torch.utils.data.DataLoader(active_set.pool,
                                    batch_size=active_learning_dict["batch_size_pool"],
                                    drop_last=False)

            n_pool = len(active_set.pool)
            score_list = torch.ones(n_pool) * -1
            pbar = tqdm.tqdm(desc="Scoring pool", total=n_pool, leave=False)
            s_ind = 0
            for batch in pool_loader:
                scores = self.score_on_batch(batch, active_learning_dict)
                n_scores = batch['images'].shape[0]

                score_list[s_ind:s_ind+n_scores] = scores.cpu()
                s_ind += n_scores

                pbar.set_description("Scoring pool")
                pbar.update(scores.shape[0])

            pbar.close()
            assert -1 not in score_list
            # higher is better
            scores, ranks = score_list.sort()
            indices = ranks[-active_learning_dict['ndata_to_label']:]

            return indices


    def mcmc_on_batch(self, batch, active_learning_dict):
        self.eval()
        al.set_dropout_train(self)

        # put images to cuda
        images = batch["images"]
        images = images.cuda()

        # variables
        n_mcmc = active_learning_dict["n_mcmc"]
        input_shape = images.size()
        batch_size = input_shape[0]

        # multiply images with n_mcmc        
        images_stacked = torch.stack([images] * n_mcmc)
        images_stacked = images_stacked.view(batch_size * n_mcmc, 
                                             *input_shape[1:])
        
        # compute the logits
        logits = self.model_base(images_stacked)
        logits = logits.view([n_mcmc, batch_size, *logits.size()[1:]])

        probs = F.softmax(logits, dim=2)

        return probs 


class ClfMonitor:
    def __init__(self):
        self.wrongs = 0
        self.n_samples = 0

    def add(self, wrongs):
        self.wrongs += wrongs.sum()
        self.n_samples += wrongs.shape[0]

    def get_avg_score(self):
        return {"val_error": (self.wrongs/ self.n_samples)}


# Architectures
# =============

class LeNeT(nn.Module):
    def __init__(self):
        super().__init__()
        nb_filters = 32
        nb_conv = 4
        self.nb_pool = 2

        self.conv1 = nn.Conv2d(1, nb_filters, (nb_conv,nb_conv),  padding=0)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters, (nb_conv, nb_conv),  padding=0)
        # self.conv3 = nn.Conv2d(nb_filters, nb_filters*2, (nb_conv, nb_conv), 1)
        # self.conv4 = nn.Conv2d(nb_filters*2, nb_filters*2, (nb_conv, nb_conv), 1)

        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(3872, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x_input):
        n,c,h,w = x_input.shape
        x = self.conv1(x_input)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, self.nb_pool, self.nb_pool)
        x =  self.dropout1(x)

        x = x.view(n, -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x