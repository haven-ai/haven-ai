"""Utils."""
import glob
import pylab as plt
import tqdm
import numpy as np
import torch
import json 
import os 
from torch.utils.data import sampler
from datetime import datetime
import pytz
import time
from skimage.color import label2rgb

def set_dropout_train(model):
    flag = False
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            flag = True
            module.train()
    assert(flag)


def mask_to_rle(binmask):
    # convert to numpy
    if not isinstance(binmask, np.ndarray):
        binmask = binmask.cpu().numpy()

    # convert to rle
    rle = maskUtils.encode(np.asfortranarray(binmask).astype("uint8"))

    return rle


def rle_to_mask(rle):
    # return a tensor in cuda
    return torch.from_numpy(maskUtils.decode(rle))

def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z.cuda(), torch.log(y))


def create_tiny(dataset, size=5):
    data = [dataset[i] for i in range(size)]

    class TinyDataset(torch.nn.Module):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, item):
            return self.data[item]

        def __len__(self):
            return len(self.data)

    dataset = TinyDataset(data)
    return dataset
    
@torch.no_grad()
def val_epoch(model, val_loader, epoch):
    model.eval()
    model.reset_val_metrics()

    n_batches = len(val_loader)
    pbar = tqdm.tqdm(desc="%d - Validating" % epoch, total=n_batches, leave=False)
    for i, batch in enumerate(val_loader):
        model.val_step(batch)
        score = model.get_val_dict()["val_score"]
        pbar.set_description("%d - Validating: %.4f" % (epoch, score))
        pbar.update(1)

    pbar.close()

    return model.get_val_dict()

def assert_dropout_exists(model):
    for name, child in model.named_modules():
        flag = False
        if isinstance(child, torch.nn.Dropout) or isinstance(child, torch.nn.Dropout2d):
            flag = True
            break
    assert flag

def set_dropout_train(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()

@torch.no_grad()
def score_pool(pool_set, model, batch_size, heuristic_name, reduction_name, epoch):
    model.eval()
    set_dropout_train(model)
    pool_loader = torch.utils.data.DataLoader(pool_set, shuffle=False, batch_size=batch_size, num_workers=0)

    score_list = torch.ones(len(pool_set)) * -1
    pbar = tqdm.tqdm(desc="%d - Scoring pool" % epoch, total=len(pool_loader), leave=False)
    s_ind = 0
    for batch in pool_loader:
        scores = heuristics.compute_heuristic_scores(model, batch,
                                                     heuristic_name=heuristic_name)
        n_scores = len(scores)

        if reduction_name == "sum":
            scores_reduced = scores.view(n_scores, -1).sum(1)
        elif reduction_name == "mean":
            scores_reduced = scores.view(n_scores, -1).mean(1)

        score_list[s_ind:s_ind+n_scores] = scores_reduced.cpu()
        s_ind += n_scores

        pbar.set_description("%d - Scoring pool" % epoch)
        pbar.update(1)

    pbar.close()
    assert -1 not in score_list

    return score_list

@torch.no_grad()
def get_probabilities_base(pool, model, n_mcmc, batch_size):
    model.eval()
    pool_loader = torch.utils.data.DataLoader(pool, shuffle=False, batch_size=batch_size)

    prob_list = []
    pbar = tqdm.tqdm(total=len(pool_loader), leave=False)
    for batch in pool_loader:
        probs = model.compute_probs(batch, n_mcmc=n_mcmc)
        prob_list += [probs.cpu().numpy()]

        pbar.set_description("Probs for active learning")
        pbar.update(1)

    pbar.close()

    prob_arr = np.vstack(prob_list)
    return prob_arr

def collate_fn(batch):
    batch_dict = {}
    for k in batch[0]:
        batch_dict[k] = []
        for i in range(len(batch)):
            
            batch_dict[k] += [batch[i][k]]
    # tuple(zip(*batch))
    return batch_dict 
    
def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

def save_json(fname, data):
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

#
# def load_latest(exp_dict, model, active_set, reset=False):
#     exp_meta = em.get_exp_meta(exp_dict)
#     history_path = exp_meta["savedir"] + "/history.pth"
#     ckp_path = exp_meta["savedir"] + "/checkpoint.pth"
#
#     if os.path.exists(ckp_path) and os.path.exists(history_path) and not reset:
#         ckp = torch.load(ckp_path)
#
#         model.load_state_dict(ckp['model_state_dict'])
#         model.opt.load_state_dict(ckp['opt_state_dict'])
#         history = mlkit_ut.load_pkl(history_path)
#         score_dict = history["score_list"][-1]
#         active_set._labelled = score_dict['labeled_data']
#
#     else:
#         print("Epoch 0: starting from scratch")
#         history = {"score_list":[]}
#
#     return model, history, active_set
#
# def save_latest(exp_dict, model, history):
#     exp_meta = em.get_exp_meta(exp_dict)
#     history_path = exp_meta["savedir"] + "/history.pth"
#     ckp_path = exp_meta["savedir"] + "/checkpoint.pth"
#
#     ckp = {"model_state_dict":model.state_dict(),
#            "opt_state_dict":model.opt.state_dict()}
#
#     torch.save(ckp, ckp_path)
#     mlkit_ut.save_pkl(history_path, history)



def get_dataloader_dict(exp_dict, train_loader):
    """Get data loader dictionary."""
    dataloader_dict = {}

    if "n_total_iters" in exp_dict["option"]:
        n_total_iters = int(exp_dict["option"]["n_total_iters"])
    else:
        n_total_iters = (len(train_loader.dataset) *
                         int(exp_dict["option"]["epochs"]))
        n_total_iters = n_total_iters / int(exp_dict["option"]["batch_size"])

    dataloader_dict["n_batches"] = len(train_loader)
    dataloader_dict["n_total_iters"] = n_total_iters

    return dataloader_dict


def load_ind_prev_images(savedir_base, exp_id):
    path = savedir_base + "/%s/ind_prev/" % exp_id

    fname_list = glob.glob(path + "*")
    for fname in fname_list:
        plt.figure()
        plt.title("%s" % fname)
        image = haven.imread(fname)
        plt.imshow(image)

def load_selected_images(savedir_base, exp_id):
    path = savedir_base + "/%s/selected/" % exp_id

    fname_list = glob.glob(path + "*")
    for fname in fname_list:
        plt.figure()
        plt.title("%s" % fname)
        image = haven.imread(fname)
        plt.imshow(image)

def load_selected_neg_images(savedir_base, exp_id):
    path = savedir_base + "/%s/selected_neg/" % exp_id

    fname_list = glob.glob(path + "*")
    for fname in fname_list:
        plt.figure()
        plt.title("%s" % fname)
        image = haven.imread(fname)
        plt.imshow(image)

import copy
def get_prev_exp_dict(exp_dict):
    exp_dict_prev = copy.deepcopy(exp_dict)
    exp_dict_prev['savedir_base'] = exp_dict_prev['savedir_base'].replace("/non_borgy/","/borgy/") 
    exp_dict_prev["sampler_dict"]["stage"] = exp_dict["sampler_dict"]["stage"] - 1
    return exp_dict_prev

def save_img_list(savedir_images, img_list):
    os.makedirs(os.path.dirname(savedir_images),exist_ok=True)
    for i, img in enumerate(img_list):
        # plt.figure(figsize=(20,30))
        plt.figure()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(savedir_images + "_%d.jpg" % i)
        # plt.show()
        plt.close()

class ExactSampler(sampler.Sampler):
    def __init__(self, train_set, indices=np.arange(5)):
        self.n_samples = len(train_set)
        self.indices = indices

    def __iter__(self):
        
        indices =  np.array(self.indices)
            
        return iter(torch.from_numpy(indices).long())

    def __len__(self):
        return len(self.indices)

def time_to_montreal():
    ts = time.time()
    utc_dt = datetime.utcfromtimestamp(ts)

    aware_utc_dt = utc_dt.replace(tzinfo=pytz.utc)

    tz = pytz.timezone('America/Montreal')
    dt = aware_utc_dt.astimezone(tz)
    dt = datetime.fromtimestamp(ts, tz)

    return dt.strftime("%I:%M %p (%b %d)")