import random

from cross.dset import CrossDataset


class OverSampler:
    def __init__(self, dset: CrossDataset, oversample: int = 2):
        self.negative_idxs = [idx for idx, label in enumerate(dset.labels) if label == 0]
        self.positive_idxs = [idx for idx, label in enumerate(dset.labels) if label == 1]
        self.oversample = oversample

    def __iter__(self):
        idxs = self.negative_idxs + self.positive_idxs * self.oversample
        random.shuffle(idxs)
        yield from idxs

    def __len__(self):
        return len(self.negative_idxs) + self.oversample * len(self.positive_idxs)
