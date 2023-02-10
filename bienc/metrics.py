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


def get_min_max_ranks(indices, data_ids: list[str], data2gold: dict[str, set[str]], e2i: dict[str, int]):
    """
    Get ranks of gold labels in INDICES gathered from predictions and Nearest Neighbor search.
    Returns both the the minimum and maximum rank of gold entities among predicted indices.
    :param indices: numpy array of shape (num_data_ids, num_neighbors) containing predicted entity indices
    :param data_ids: data ids in the same order as indices
    :param data2gold: dict mapping data id to set of topic ids
    :param e2i: dict mapping entity id to topic index
    :return: numpy array of shape (num_data_ids,) with lowest rank of gold label, -1 if not found
    """
    min_ranks = np.full(indices.shape[0], -1, dtype=float)
    max_ranks = np.full(indices.shape[0], -1, dtype=float)
    i = 0
    for idxs, data_id in zip(indices, data_ids):
        gold = data2gold[data_id]
        gold_idxs = np.array([e2i[g] for g in gold])
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
