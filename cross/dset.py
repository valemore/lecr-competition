from typing import Iterable, Dict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from bienc.tokenizer import tokenize


class PositivesNegativesDataset(Dataset):
    """Dataset with positives and negatives to be used for crossencoder training."""
    def __init__(self,
                 topic_ids: Iterable[str],
                 topic_positive_ids: Iterable[Iterable[str]],
                 topic_negative_ids: Iterable[Iterable[str]],
                 topic2text: Dict[str, str], content2text: Dict[str, str],
                 topic_num_tokens: int, content_num_tokens: int):
        """
        Training dataset for the bi-encoder embedding content and topic texts into the same space.
        :param topic_ids: iterable over topic ids
        :param topic_positive_ids: iterable of iterable over positive content ids
        :param topic_negative_ids: iterable of iterable over negative content ids
        :param topic2text: dictionary mapping topic to its text representation
        :param content2text: dictionary mapping content to its text representation
        :param topic_num_tokens: how many tokens to use for topic representation
        :param content_num_tokens: how many tokens to use for content representation
        """
        self.topic_ids = []
        self.content_ids = []
        self.labels = []
        self.topic2text = topic2text
        self.content2text = content2text
        self.topic_num_tokens = topic_num_tokens
        self.content_num_tokens = content_num_tokens

        for topic_id, positive_ids, negative_ids, in tqdm(zip(topic_ids, topic_positive_ids, topic_negative_ids)):
            for content_id in positive_ids.split():
                self.topic_ids.append(topic_id)
                self.content_ids.append(content_id)
                self.labels.append(1)
            for content_id in negative_ids.split():
                self.topic_ids.append(topic_id)
                self.content_ids.append(content_id)
                self.labels.append(0)

    def __getitem__(self, idx):
        topic_encoded = tokenize(self.topic2text[self.topic_ids[idx]], self.topic_num_tokens)
        content_encoded = tokenize(self.content2text[self.content_ids[idx]], self.content_num_tokens)
        return torch.tensor(topic_encoded["input_ids"]), torch.tensor(topic_encoded["attention_mask"]), \
               torch.tensor(content_encoded["input_ids"]), torch.tensor(content_encoded["attention_mask"]), \
               self.labels[idx]

    def __len__(self):
        return len(self.topic_ids)
