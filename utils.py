# Utility functions
import os
import pickle
import random
from collections import defaultdict, OrderedDict
from typing import Union, Callable, Any, Tuple, Dict, List, Set

import numpy as np
import pandas as pd
import torch

from typehints import FName, StateDict


def seed_everything(seed):
    """
    Makes results as reproducible as possible.
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True # Choose deterministic algorithms
        torch.backends.cudnn.benchmark = False    # Don't benchmark algorithms and choose fastest
        torch.backends.cudnn.enabled = True


def get_learning_rate_momentum(optimizer: torch.optim.Optimizer) -> Tuple[float, Union[float, None]]:
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


def flatten_content_ids(corr_df: pd.DataFrame) -> List[str]:
    """Get flat list of all content ids in the correlation DataFrame."""
    return sorted(list(set([content_id for content_ids in corr_df["content_ids"] for content_id in content_ids.split()])))


def flatten_positive_negative_content_ids(corr_df: pd.DataFrame) -> List[str]:
    """Get flat list of all positive and negative content ids in the correlation DataFrame."""
    return sorted(list(set([content_id for content_ids in corr_df["content_ids"] for content_id in content_ids.split()] + [content_id for content_ids in corr_df["cands"] for content_id in content_ids.split()])))


def get_content_id_gold(corr_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Get dictionary mapping content id to set of correct topic ids."""
    c2gold = defaultdict(set)
    for topic_id, content_ids in zip(corr_df["topic_id"], corr_df["content_ids"]):
        for content_id in content_ids.split():
            c2gold[content_id].add(topic_id)
    return c2gold


def get_topic_id_gold(corr_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Get dictionary mapping topic id to set of correct content ids."""
    t2gold = {}
    for topic_id, content_ids in zip(corr_df["topic_id"], corr_df["content_ids"]):
        t2gold[topic_id] = set()
        assert len(content_ids) > 0
        for content_id in content_ids.split():
            t2gold[topic_id].add(content_id)
    return t2gold


def get_t2lang_c2lang(corr_df: pd.DataFrame, content_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    t2lang = {}
    for topic_id, language in zip(corr_df["topic_id"], corr_df["language"]):
        t2lang[topic_id] = language
    c2lang = {}
    for content_id, language in zip(content_df["id"], content_df["language"]):
        c2lang[content_id] = language
    return t2lang, c2lang


def is_ordered(data_ids: List[str]) -> bool:
    """Verifies whether DATA_IDS are ordered."""
    if len(data_ids) < 1:
        return True
    prev = data_ids[0]
    for next in data_ids[1:]:
        if prev > next:
            return False
    return True


def are_entity_ids_aligned(entity_ids: List[str], e2i: Dict[str, int]) -> bool:
    """Verifies whether ENTITY_IDS and E2I represent same order of entities."""
    if len(entity_ids) != len(e2i):
        return False
    for entity_idx, entity_id in enumerate(entity_ids):
        if e2i[entity_id] != entity_idx:
            return False
    return True


def sanity_check_inputs(content_df, corr_df, topics_df):
    content_ids_set = set(content_df["id"])
    assert len(content_df["id"]) == len(content_ids_set)
    assert is_ordered(content_df["id"])
    assert len(corr_df["topic_id"]) == len(set(corr_df["topic_id"]))
    assert is_ordered(corr_df["topic_id"])
    topic_ids_set = set(topics_df["id"])
    assert len(topics_df["id"]) == len(topic_ids_set)
    assert is_ordered(topics_df["id"])

    for topic_id, content_ids in zip(corr_df["topic_id"], corr_df["content_ids"]):
        assert topic_id in topic_ids_set
        assert len(content_ids) > 0
        for content_id in content_ids.split():
            assert content_id in content_ids_set


def safe_div(num, den):
    """Returns 0.0 if den is zero. Used for precision computation."""
    if den == 0.0:
        return 0.0
    return float(num) / float(den)


def safe_div_np(num, den):
    """Safe division for numpy arrays. Output has same shape as inputs and is zero where denominator is zero. Used for precision computation."""
    mask = den == 0.0
    den[mask] = 1.0
    out = num / den
    out[mask] = 0.0
    return out


def state_dict_to(state_dict, device: torch.device):
    out = OrderedDict()
    for k, v in state_dict.items():
        out[k] = v.to(device)
    return out
