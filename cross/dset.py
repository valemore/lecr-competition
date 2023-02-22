from typing import Iterable, Dict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from bienc.tokenizer import tokenize_cross


class PositivesNegativesDataset(Dataset):
    """Dataset with positives and negatives to be used for crossencoder training."""
    def __init__(self,
                 topic_ids: Iterable[str],
                 topic_positive_ids: Iterable[str],
                 topic_negative_ids: Iterable[str],
                 topic2text: Dict[str, str], content2text: Dict[str, str],
                 num_tokens: int):
        """
        Training dataset for the bi-encoder embedding content and topic texts into the same space.
        :param topic_ids: iterable over topic ids
        :param topic_positive_ids: iterable over concatenated positive content ids
        :param topic_negative_ids: iterable over concatenated negative content ids
        :param topic2text: dictionary mapping topic to its text representation
        :param content2text: dictionary mapping content to its text representation
        :param num_tokens: how many tokens to use for joint representation
        """
        self.topic_ids = []
        self.content_ids = []
        self.labels = []
        self.topic2text = topic2text
        self.content2text = content2text
        self.num_tokens = num_tokens

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
        enc = tokenize_cross(self.topic2text[self.topic_ids[idx]], self.content2text[self.content_ids[idx]], self.num_tokens)
        return torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"]), self.labels[idx]

    def __len__(self):
        return len(self.topic_ids)


class CrossInferenceDataset(Dataset):
    """Dataset for crossencoder inference."""
    def __init__(self,
                 topic_ids: Iterable[str],
                 topic_cand_ids: Iterable[Iterable[str]],
                 topic2text: Dict[str, str], content2text: Dict[str, str],
                 num_tokens: int):
        """
        Training dataset for the bi-encoder embedding content and topic texts into the same space.
        :param topic_ids: iterable over topic ids
        :param topic_cand_ids: iterable over concatenated candidate ids
        :param topic2text: dictionary mapping topic to its text representation
        :param content2text: dictionary mapping content to its text representation
        :param num_tokens: how many tokens to use for joint representation
        """
        self.topic_ids = []
        self.content_ids = []
        self.topic2text = topic2text
        self.content2text = content2text
        self.num_tokens = num_tokens

        for topic_id, cand_ids in tqdm(zip(topic_ids, topic_cand_ids)):
            for content_id in cand_ids.split():
                self.topic_ids.append(topic_id)
                self.content_ids.append(content_id)

    def __getitem__(self, idx):
        enc = tokenize_cross(self.topic2text[self.topic_ids[idx]], self.content2text[self.content_ids[idx]], self.num_tokens)
        return torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"])

    def __len__(self):
        return len(self.topic_ids)
