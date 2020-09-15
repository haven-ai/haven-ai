import torch
import torchvision
import os
import re
import time
import zipfile
import shutil
import tqdm
import random
from torch.utils.data import sampler
from PIL import Image
from itertools import cycle, islice


# ========================================================
# Dataset-related functions and classes
# ========================================================

def subset_dataset(dataset, size):
    data = [dataset[i] for i in range(size)]
    class SubsetDataset(torch.nn.Module):
        def __init__(self, data, dataset):
            self.data = data
            self.split = dataset.split

        def __getitem__(self, item):
            return self.data[item]

        def __len__(self):
            return len(self.data)

    dataset = SubsetDataset(data, dataset)
    return dataset

def get_indices(dataset, class_indices):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_indices:
            indices.append(i)
    return indices

def get_indices_unique(dataset, class_indices):
    indices = []
    obtained_class_indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_indices\
            and dataset.targets[i] not in obtained_class_indices:
            indices.append(i)
            obtained_class_indices.append(dataset.targets[i])
    return indices

class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, shuffle_cond=True):
        self.dataset = dataset
        self.targets = self.dataset.targets
        classes_to_indices = list(range(len(self.dataset.classes)))
        self.dataset_class_split = {
            classes_to_indices[i]: get_indices(self.dataset, 
                                               [classes_to_indices[i]]) 
            for i in classes_to_indices
        }
        self.shuffle_cond = shuffle_cond

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.shuffle_cond:
            cond_index = random.choice(self.dataset_class_split[target])
        else:
            cond_index = index
        cond_image, cond_target = self.dataset[cond_index]
        return img, target, cond_image, cond_target

    def __len__(self):
        return len(self.dataset)

# ========================================================
# Image utility functions
# ========================================================

def reformat_image(filename, data):
    pass

def reformat_images(images):
    pass

def stack_img_list(img_list):
    image_list_torch = []
    for i in img_list:
        if i.ndim == 4:
            i = i[0]
        if i.max() > 1:
            i = i / 255.
        image_list_torch += [i]

    image_list_torch = torch.stack(image_list_torch)
    img = torchvision.utils.make_grid(image_list_torch, nrow=5)
    return img

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    img = data.detach().clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def save_images(save_dir, images, epoch=None, batch_id=None, filenames=None):
    i = 0
    for image in images:
        filename = str(i).zfill(8) + '.png'
        if epoch is not None and batch_id is not None and filenames is None:
            epoch_str = str(epoch).zfill(8)
            batch_id_str = str(batch_id).zfill(8)
            filename = epoch_str + '_' + batch_id_str + '_' + filename
        if filenames is not None:
            filename = filenames[i]
        save_path = os.path.join(save_dir, filename)
        save_image(save_path, image)
        i += 1

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div(255.0)
    return (batch - mean) / std

def unnormalize_batch(batch):
    # unnormalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return ((batch * std) + mean) * 255.0

# ========================================================
# Misc.
# ========================================================

def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        zf.extractall(path=dest_dir)

def load_state_dict(fname_model, style_model):
    state_dict = torch.load(fname_model)

    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]

    style_model.load_state_dict(state_dict)
    style_model.cuda()
    return style_model

def rmtree(dir):
    shutil.rmtree(dir, ignore_errors=True)

'''
MIT license: http://opensource.org/licenses/MIT
Copyright (c) <2013> <Paulo Cheque>
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

def flatten_list(alist):
    '''
    No Python hacks in this implementation. Also, this accepts many levels of nested lists.
    The limit is in the number of recursive calls.
    @alist: A tuple or list.
    @return: A flat list with all elements of @alist and its nested lists.
    Complexity: `Θ(n)`, where `n` is the number of elements of @alist
    plus the number of elements of all nested lists.
    '''
    new_list = []
    for item in alist:
        if isinstance(item, (list, tuple)):
            new_list.extend(flatten_list(item))
        else:
            new_list.append(item)
    return new_list


def add_hparams(self, hparam_dict, metric_dict, global_step=None):
    from torch.utils.tensorboard.summary import hparams
    """Add a set of hyperparameters to be compared in TensorBoard.
    Args:
        hparam_dict (dictionary): Each key-value pair in the dictionary is the
            name of the hyper parameter and it's corresponding value.
        metric_dict (dictionary): Each key-value pair in the dictionary is the
            name of the metric and it's corresponding value. Note that the key used
            here should be unique in the tensorboard record. Otherwise the value
            you added by `add_scalar` will be displayed in hparam plugin. In most
            cases, this is unwanted.

        p.s. The value in the dictionary can be `int`, `float`, `bool`, `str`, or
        0-dim tensor
    Examples::
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter() as w:
            for i in range(5):
                w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
    Expected result:
    .. image:: _static/img/tensorboard/add_hparam.png
        :scale: 50 %
    """
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError('hparam_dict and metric_dict should be dictionary.')
    exp, ssi, sei = hparams(hparam_dict, metric_dict)

    self.file_writer.add_summary(exp, global_step)
    self.file_writer.add_summary(ssi, global_step)
    self.file_writer.add_summary(sei, global_step)
    for k, v in metric_dict.items():
        self.add_scalar(k, v, global_step)


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))