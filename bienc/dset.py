# PyTorch datasets for the Bi-encoder
from collections.abc import Iterable
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenizer import tokenizer


def tokenize(text: str, num_tokens: int):
    """
    Get input ids and attention mask.
    :param text: text to tokenize
    :param num_tokens: truncate and pad to this many tokens
    :return: dict with input ids and attention mask
    """
    enc = tokenizer(text,
                    max_length=num_tokens,
                    truncation="only_first",
                    padding="max_length",
                    add_special_tokens=True,
                    return_overflowing_tokens=False,
                    return_offsets_mapping=False)

    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}


class BiencDataset(Dataset):
    """Biencoder dataset to be used for training."""
    def __init__(self,
                 topic_ids: Iterable[str], topic_content_ids: Iterable[Iterable[str]],
                 topic2text: Dict[str, str], content2text: Dict[str, str],
                 topic_num_tokens: int, content_num_tokens: int,
                 t2i: Dict[str, int]):
        """
        Training dataset for the bi-encoder embedding content and topic texts into the same space.
        :param topic_ids: iterable over topic ids
        :param topic_content_ids: iterable of iterable over content ids
        :param topic2text: dictionary mapping topic to its text representation
        :param content2text: dictionary mapping content to its text representation
        :param t2i: dictionary mapping topic id to index
        """
        self.topic_ids = []
        self.content_ids = []
        self.topic2text = topic2text
        self.content2text = content2text
        self.topic_num_tokens = topic_num_tokens
        self.content_num_tokens = content_num_tokens
        self.t2i = t2i
        self.i2t = {idx: topic_id for topic_id, idx in self.t2i.items()}

        for topic_id, content_ids, topic_id in tqdm(zip(topic_ids, topic_content_ids, topic_ids)):
            for content_id in content_ids.split():
                self.topic_ids.append(topic_id)
                self.content_ids.append(content_id)

    def __getitem__(self, idx):
        topic_encoded = tokenize(self.topic2text[self.topic_ids[idx]], self.topic_num_tokens)
        content_encoded = tokenize(self.content2text[self.content_ids[idx]], self.content_num_tokens)
        return torch.tensor(topic_encoded["input_ids"]), torch.tensor(topic_encoded["attention_mask"]), \
               torch.tensor(content_encoded["input_ids"]), torch.tensor(content_encoded["attention_mask"]), \
               self.t2i[self.topic_ids[idx]]

    def __len__(self):
        return len(self.topic_ids)


class BiencInferenceDataset(Dataset):
    """Biencoder dataset for the topics and contents to be used during inference."""
    def __init__(self, data_ids: List[str], data2text: Dict[str, str], num_tokens: int):
        """
        Inference dataset for the bi-encoder embedding content and topic texts into the same space.
        :param data_ids: iterable over topic ids or content ids
        :param data2text: dictionary mapping topic or content to its text representation
        """
        self.content_ids = data_ids
        self.content2text = data2text
        self.num_tokens = num_tokens

    def __getitem__(self, idx):
        content_encoded = tokenize(self.content2text[self.content_ids[idx]], self.num_tokens)
        return torch.tensor(content_encoded["input_ids"]), torch.tensor(content_encoded["attention_mask"])

    def __len__(self):
        return len(self.content_ids)
