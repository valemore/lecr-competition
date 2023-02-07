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


def get_recall_dct(scores, labels):
    idxs = np.argsort(-scores, axis=1)
    rank = np.argwhere(idxs == np.expand_dims(labels, -1))[:, 1]
    recall_dct = {
        1: 0.0,
        3: 0.0,
        5: 0.0,
        10: 0.0,
        20: 0.0
    }
    for at in recall_dct:
        recall_dct[at] = np.sum(rank < at) / len(rank)
    return recall_dct


def get_ranks(scores, labels):
    """Get ranks of gold LABELS in SCORES."""
    idxs = np.argsort(-scores, axis=1)
    rank = np.argwhere(idxs == np.expand_dims(labels, -1))[:, 1]
    return rank


def log_recall_dct(recall_dct, global_step, run, label):
    """Log a recall dictionary to neptune.ai"""
    for k, v in recall_dct.items():
        run[f"{label}/recall@{k}"].log(v, step=global_step)
