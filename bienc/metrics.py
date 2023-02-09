# Metrics for the Bi-encoder
import numpy as np


def get_recall_dct(ranks) -> dict[int, float]:
    """Get recall dictionary from RANKS given as numpy array of shape (num_examples,)."""
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


def get_min_max_ranks(indices, flat_content_ids: list[str], c2gold: dict[str, set[str]], t2i: dict[str, int]):
    """
    Get ranks of gold labels in INDICES gathered from predictions and Nearest Neighbor search.
    Returns both the the minimum and maximum rank of gold topics among predicted indices.
    :param indices: numpy array of shape (num_content_ids, num_neighbors) containing predicted topic indices
    :param flat_content_ids: content ids in the same order as indices
    :param c2gold: dict mapping content it to set of topic ids
    :param t2i: dict mapping topic id to topic index
    :return: numpy array of shape (num_content_ids,) with lowest rank of gold label, -1 if not found
    """
    min_ranks = np.full(indices.shape[0], -1, dtype=float)
    max_ranks = np.full(indices.shape[0], -1, dtype=float)
    i = 0
    for idxs, content_id in zip(indices, flat_content_ids):
        gold = c2gold[content_id]
        gold_idxs = np.array([t2i[g] for g in gold])
        found = np.argwhere(idxs.reshape(-1, 1) == gold_idxs.reshape(1, -1))[:, 0]
        if len(found) > 0:
            min_ranks[i] = min(found)
            max_ranks[i] = max(found)
        else:
            min_ranks[i] = np.inf
            max_ranks[i] = np.inf
        i += 1
    return min_ranks, max_ranks


def get_mean_inverse_rank(ranks) -> float:
    """Compute mean inverse rank for RANKS of shape (num_examples,)."""
    mir = 1.0 / (ranks + 1)
    return np.mean(mir).item()