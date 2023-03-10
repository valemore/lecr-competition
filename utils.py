# Utility functions
import os
import pickle
import random
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Union, Callable, Any, Tuple, Dict, List, Set

import numpy as np
import pandas as pd
import torch

from config import CFG
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


def get_learning_rate_momentum(optimizer: torch.optim.Optimizer) -> Tuple[float, Union[float, None], Union[float, None]]:
    """Get learning rate and momentum for PyTorch optimizer OPTIMIZER."""
    lr = optimizer.param_groups[0]["lr"]
    momentum = optimizer.param_groups[0].get("momentum", None)
    if CFG.head_lr:
        classifier_lr = optimizer.param_groups[1]["lr"]
    else:
        classifier_lr = None
    return lr, momentum, classifier_lr


def save_checkpoint(fname: FName, global_step: int,
                    model_state_dict: StateDict, optimizer_state_dict: StateDict, scheduler_state_dict: StateDict, scaler_state_dict: StateDict) -> None:
    """Save dictionary of model and training state to disk."""
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'global_step': global_step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'scaler_state_dict': scaler_state_dict,
    }, fname)


def sanitize_fname(fname: str) -> str:
    """Sanitize string for including it in file name."""
    return fname.replace("/", "-")


def flatten_content_ids(df: pd.DataFrame) -> List[str]:
    """Get flat list of all content ids in the input DataFrame."""
    return sorted(list(set([content_id for content_ids in df["content_ids"] for content_id in content_ids.split()])))


def get_content_id_gold(input_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Get dictionary mapping content id to set of correct topic ids."""
    c2gold = defaultdict(set)
    for topic_id, content_ids in zip(input_df["topic_id"], input_df["content_ids"]):
        for content_id in content_ids.split():
            c2gold[content_id].add(topic_id)
    return c2gold


def get_topic_id_gold(input_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Get dictionary mapping topic id to set of correct content ids."""
    t2gold = {}
    for topic_id, content_ids in zip(input_df["topic_id"], input_df["content_ids"]):
        t2gold[topic_id] = set()
        assert len(content_ids) > 0
        for content_id in content_ids.split():
            t2gold[topic_id].add(content_id)
    return t2gold


def get_t2lang_c2lang(input_df: pd.DataFrame, content_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    t2lang = {}
    for topic_id, language in zip(input_df["topic_id"], input_df["language"]):
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


def are_content_ids_aligned(content_ids: List[str], c2i: Dict[str, int]) -> bool:
    """Verifies whether CONTENT_IDS and C2I represent the same content ids."""
    check = c2i.copy()
    if "dummy" in check:
        check.pop("dummy")
    if len(content_ids) != len(check):
        return False
    for entity_idx, entity_id in enumerate(content_ids):
        if check[entity_id] != entity_idx:
            return False
    return True


def safe_div(a, b):
    if b == 0.0:
        return 0.0
    return a / b


def safe_div_np(num, den):
    """Safe division for numpy arrays. Output has same shape as inputs and is zero where denominator is zero. Used for precision computation."""
    mask = den == 0.0
    den[mask] = 1.0
    out = num / den
    out[mask] = 0.0
    return out


def get_dfs(data_dir: FName, mode: str):
    assert mode in ("bienc", "cross", "submit")

    def fix_legacy_cands(cat_cand_ids: str):
        return " ".join([cand_id for cand_id in cat_cand_ids.split() if cand_id != "dummy"])

    def truncate_cands(cat_cand_ids: str, max_num_cands):
        return " ".join([cand_id for cand_id in cat_cand_ids.split()][:max_num_cands])

    data_dir = Path(data_dir)
    topics_df = pd.read_csv(data_dir / "topics.csv", keep_default_na=False)
    topics_df["title"] = topics_df["title"].str.strip()
    topics_df["description"] = topics_df["description"].str.strip()
    content_df = pd.read_csv(data_dir / "content.csv", keep_default_na=False)
    content_df["title"] = content_df["title"].str.strip()
    content_df["description"] = content_df["description"].str.strip()
    content_df["text"] = content_df["text"].str.strip()

    if mode == "submit":
        input_df = pd.read_csv(data_dir / "sample_submission.csv", keep_default_na=False)
        input_df = input_df.merge(topics_df.loc[:, ["id", "language"]], left_on="topic_id", right_on="id", how="left")
    elif mode == "bienc":
        input_df = pd.read_csv(data_dir / "correlations.csv", keep_default_na=False)
        input_df = input_df.merge(topics_df.loc[:, ["id", "language"]], left_on="topic_id", right_on="id", how="left").drop(columns=["id"])
    else:
        input_df = pd.read_csv(CFG.cross_corr_fname, keep_default_na=False)
        # Compatibility with old generated cross dfs
        input_df["cands"] = [fix_legacy_cands(cands) for cands in input_df["cands"]]
        input_df["cands"] = [truncate_cands(cands, CFG.cross_num_cands) for cands in input_df["cands"]]

    # Sort
    topics_df = topics_df.sort_values("id").reset_index(drop=True)
    content_df = content_df.sort_values("id").reset_index(drop=True)
    input_df = input_df.sort_values("topic_id").reset_index(drop=True)

    return topics_df, content_df, input_df


def get_content_ids_c2i(content_df: pd.DataFrame):
    content_ids = sorted(list(set(content_df["id"])))
    c2i = {content_id: content_idx for content_idx, content_id in enumerate(content_ids)}
    return content_ids, c2i


def merge_cols(df: pd.DataFrame, other: pd.DataFrame, cols: List[str],
               on: Union[None, str] = None, left_on: Union[None, str] = None, right_on: Union[None, str] = None):
    """Merges COLS from OTHER into dataframe DF and drops duplicate id columns."""
    if on is not None:
        assert left_on is None and right_on is None
        kept_cols = [on] + cols
        return df.merge(other.loc[:, kept_cols], on=on, how="left")
    if left_on is not None or right_on is not None:
        assert left_on is not None and right_on is not None and on is None
        kept_cols = [right_on] + cols
        return df.merge(other.loc[:, kept_cols], left_on=left_on, right_on=right_on, how="left").drop(columns=[right_on])
