# Utility functions
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_learning_rate_momentum(optimizer):
    """Get learning rate and momentum for PyTorch optimizer OPTIMIZER."""
    pg_idx = max([idx for idx in range(len(optimizer.param_groups))], key=lambda idx: optimizer.param_groups[idx]["lr"])
    lr = optimizer.param_groups[pg_idx]["lr"]
    momentum = optimizer.param_groups[pg_idx].get("momentum", None)
    return lr, momentum


def save_checkpoint(fname, global_step, model_state_dict, optimizer_state_dict, scheduler_state_dict, scaler_state_dict):
    """Save dictionary of model and training state to disk."""
    fname.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'global_step': global_step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'scaler_state_dict': scaler_state_dict,
    }, fname)


def sanitize_model_name(model_name):
    """Sanitize model name for including it in file name."""
    return model_name.replace("/", "-")


def cache(fname, fn, refresh=False):
    """Helper to cache prepared PyTorch datasets."""
    fname.parent.mkdir(exist_ok=True, parents=True)
    if fname.exists() and not refresh:
        with open(fname, "rb") as f:
            x = pickle.load(f)
    else:
        x = fn()
        with open(fname, "wb") as f:
            pickle.dump(x, f)
    return x


def get_recall_dct(ranks):
    recall_dct = {
        1: 0.0,
        3: 0.0,
        5: 0.0,
        10: 0.0,
        20: 0.0,
        50: 0.0,
        100: 0.0
    }
    for at in recall_dct:
        recall_dct[at] = np.sum(ranks < at) / len(ranks)
    return recall_dct


def get_ranks(indices, flat_content_ids, c2gold, t2i):
    """Get ranks of gold labels in INDICES gathered from predictions and Nearest Neighbor search.
    :param indices: numpy array of shape (num_content_ids, num_neighbors) containing predicted topic indices
    :param flat_content_ids: content ids in the same order as indices
    :param c2gold: dict mapping content it to set of topic ids
    :param t2i: dict mapping topic id to topic index
    :return: numpy array of shape (num_content_ids,) with lowest rank of gold label, -1 if not found
    """
    ranks = np.full(indices.shape[0], -1, dtype=int)
    i = 0
    for idxs, content_id in zip(indices, flat_content_ids):
        gold = c2gold[content_id]
        gold_idxs = np.array([t2i[g] for g in gold])
        found = np.argwhere(idxs.reshape(-1, 1) == gold_idxs.reshape(1, -1))[:, 0]
        if len(found) > 0:
            ranks[i] = min(found)
        else:
            ranks[i] = -1
        i += 1
    return ranks


def get_mean_inverse_rank(ranks):
    notfound = ranks < 0
    mir = np.array(ranks)
    mir[notfound] = 0
    mir = 1.0 / (mir + 1)
    mir[notfound] = 0
    return np.mean(mir)


def log_recall_dct(recall_dct, global_step, run, label):
    """Log a recall dictionary to neptune.ai"""
    for k, v in recall_dct.items():
        run[f"{label}/recall@{k}"].log(v, step=global_step)
