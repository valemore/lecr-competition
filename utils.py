# Utility functions
import pickle
from collections import defaultdict
from typing import Union, Callable, Any

import pandas as pd
import torch
from neptune.new import Run

from typehints import FName, StateDict


def get_learning_rate_momentum(optimizer: torch.optim.Optimizer) -> tuple[float, Union[float, None]]:
    """Get learning rate and momentum for PyTorch optimizer OPTIMIZER."""
    pg_idx = max([idx for idx in range(len(optimizer.param_groups))], key=lambda idx: optimizer.param_groups[idx]["lr"])
    lr = optimizer.param_groups[pg_idx]["lr"]
    momentum = optimizer.param_groups[pg_idx].get("momentum", None)
    return lr, momentum


def save_checkpoint(fname: FName, global_step: int,
                    model_state_dict: StateDict, optimizer_state_dict: StateDict, scheduler_state_dict: StateDict, scaler_state_dict: StateDict) -> None:
    """Save dictionary of model and training state to disk."""
    fname.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'global_step': global_step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'scaler_state_dict': scaler_state_dict,
    }, fname)


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for including it in file name."""
    return model_name.replace("/", "-")


def cache(fname: FName, fn: Callable[[], Any], refresh: bool = False) -> Any:
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


def log_recall_dct(recall_dct: dict[int, float], global_step: int, run: Run, label: str) -> None:
    """Log a recall dictionary to neptune.ai"""
    for k, v in recall_dct.items():
        run[f"{label}/recall@{k}"].log(v, step=global_step)


def flatten_content_ids(corr_df: pd.DataFrame) -> list[str]:
    """Get flat list of all content ids in the correlation DataFrame."""
    return sorted(list(set([content_id for content_ids in corr_df["content_ids"] for content_id in content_ids.split()])))


def get_content_id_gold(corr_df: pd.DataFrame) -> dict[str, set[str]]:
    """Get dictionary mapping content id to set of correct topic ids."""
    c2gold = defaultdict(set)
    for topic_id, content_ids in zip(corr_df["topic_id"], corr_df["content_ids"]):
        for content_id in content_ids.split():
            c2gold[content_id].add(topic_id)
    return c2gold
