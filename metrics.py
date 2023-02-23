from typing import Dict, Set


def precision(gold: Set[str], pred: Set[str]):
    if len(pred) == 0:
        return 0.0
    both = {p for p in pred if p in gold}
    return len(both) / len(pred)


def recall(gold: Set[str], pred: Set[str]):
    if len(gold) == 0:
        return 0.0
    both = {g for g in gold if g in pred}
    return len(both) / len(gold)


def fscore_from_prec_rec(prec, rec, beta=2.0):
    den = (beta ** 2 * prec) + rec
    if den == 0.0:
        return 0.0
    return (1 + beta ** 2) * prec * rec / den


def single_fscore(gold, pred, beta=2.0):
    prec = precision(gold, pred)
    rec = recall(gold, pred)
    return fscore_from_prec_rec(prec, rec, beta)


def np_fscore(prec, rec, beta=2.0):
    den = (beta ** 2 * prec) + rec
    mask = den == 0
    den[mask] = 1.0
    out = (1 + beta ** 2) * prec * rec / den
    out[mask] = 0.0
    return out


def get_fscore(t2gold: Dict[str, Set[str]], t2pred: Dict[str, Set[str]], beta: float = 2.0):
    assert len(t2gold) == len(t2pred)
    score = 0.0
    for topic_id, gold in t2gold.items():
        pred = t2pred[topic_id]
        score += single_fscore(gold, pred, beta)
    return score / len(t2gold)
