import contextlib
import copy
import hashlib
import itertools
import json


import pprint
import os
import pickle
import shlex
import subprocess
import threading
import time
import numpy as np
import pylab as plt
import torch


def save_json(fname, data, makedirs=True):
    """Save data into a json file.

    Parameters
    ----------
    fname : str
        Name of the json file
    data : [type]
        Data to save into the json file
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def load_mat(fname):
    """Load a matlab file.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    dict
        Dictionary with the loaded data
    """
    import scipy.io as io

    return io.loadmat(fname)


def load_json(fname, decode=None):  # TODO: decode???
    """Load a json file.

    Parameters
    ----------
    fname : str
        Name of the file
    decode : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        Content of the file
    """
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d


def read_text(fname):
    """Loads the content of a text file.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    list
        Content of the file. List containing the lines of the file
    """
    with open(fname, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return lines


def load_pkl(fname):
    """Load the content of a pkl file.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    [type]
        Content of the file
    """
    with open(fname, "rb") as f:
        return pickle.load(f)


def save_pkl(fname, data, with_rename=True, makedirs=True):
    """Save data in pkl format.

    Parameters
    ----------
    fname : str
        File name
    data : [type]
        Data to save in the file
    with_rename : bool, optional
        [description], by default True
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    # Create folder
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)

    # Save file
    if with_rename:
        fname_tmp = fname + "_tmp.pth"
        with open(fname_tmp, "wb") as f:
            pickle.dump(data, f)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(fname_tmp, fname)
    else:
        with open(fname, "wb") as f:
            pickle.dump(data, f)
