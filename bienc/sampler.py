import math
import random

import pandas as pd
from torch.utils.data import BatchSampler

from bienc.dset import BiencDataset


def get_batches(topic_langs, batch_size, drop_last):
    idxs = list(range(len(topic_langs)))
    df = pd.DataFrame({"idx": idxs, "topic_lang": topic_langs})
    df = df.sample(frac=1).sort_values("topic_lang").reset_index(drop=True)
    batches = list(BatchSampler(df["idx"], batch_size=batch_size, drop_last=drop_last))
    random.shuffle(batches)
    return batches


class SameLanguageSampler:
    def __init__(self, dset: BiencDataset, batch_size: int, drop_last=False):
        self.topic_langs = dset.topic_langs
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = get_batches(self.topic_langs, self.batch_size, self.drop_last)
        yield from batches

    def __len__(self):
        return math.ceil(len(self.topic_langs) / self.batch_size)
