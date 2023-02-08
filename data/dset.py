# PyTorch datasets for the Bi-encoder
import logging
from collections.abc import Iterable
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tokenizer import tokenizer


logger = logging.getLogger(__name__)


def tokenize(tokenizer, text: str, num_tokens: int):
    """
    Get input ids and attention mask.
    :param tokenizer: tokenizer to use
    :param text: search query text to encode
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
        topic_encoded = tokenize(tokenizer, self.topic2text[self.topic_ids[idx]], self.topic_num_tokens)
        content_encoded = tokenize(tokenizer, self.content2text[self.content_ids[idx]], self.content_num_tokens)
        return torch.tensor(topic_encoded["input_ids"]), torch.tensor(topic_encoded["attention_mask"]), \
               torch.tensor(content_encoded["input_ids"]), torch.tensor(content_encoded["attention_mask"]), \
               self.t2i[self.topic_ids[idx]]

    def __len__(self):
        return len(self.topic_ids)


class BiencDataSetInference(Dataset):
    """Biencoder dataset for the queries to be used during inference."""
    def __init__(self, content_ids: List[str], content2text: Dict[str, str],  num_tokens: int):
        """
        Inference dataset for the bi-encoder embedding content and topic texts into the same space.
        :param content_ids: iterable over content ids
        :param content2text: dictionary mapping content to its text representation
        """
        self.content_ids = content_ids
        self.content2text = content2text
        self.num_tokens = num_tokens

    def __getitem__(self, idx):
        content_encoded = tokenize(tokenizer, self.content2text[self.content_ids[idx]], self.num_tokens)
        return torch.tensor(content_encoded["input_ids"]), torch.tensor(content_encoded["attention_mask"])

    def __len__(self):
        return len(self.content_ids)


class BiencTopicEmbeddings(Dataset):
    """Biencoder dataset for the topics to be used during inference."""
    def __init__(self, t2i, embs):
        """
        Dataset containing topic embeddings for the bi-encoder embedding content and topic texts into the same space.
        :param t2i: dictionary mapping topic id to index
        :param embs: the corresponding embeddings that were obtained by training the biencoder
        """
        self.t2i = t2i
        self.i2t = {idx: topic_id for topic_id, idx in self.t2i.items()}
        self.embs = embs

    def __getitem__(self, idx):
        return self.embs[idx, :]

    def get(self, topic_id):
        return self.embs[self.t2i[topic_id], :]

    def __len__(self):
        return len(self.t2i)

    def get_embs(self):
        return self.embs

    @classmethod
    def from_model(cls, encoder, device, data_loader, i2t: Dict[int, str]):
        """
        Initialize topic embeddings from a trained model.
        :param encoder: trained encoder of the bi-encoder
        :param device: device on which to compute embeddings
        :param data_loader: data loader of a BiencDataSetInference over the topics
        :param i2t: dictionary mapping index to topic iod
        :return: dataset containing topic embeddings
        """
        embs = []

        logger.info("Preparing Bi-encoder inference dataset containing topic embeddings...")
        encoder.eval()
        encoder.to(device)

        for batch in tqdm(data_loader):
            batch = tuple(x.to(device) for x in batch)
            with torch.no_grad():
                entity_emb = encoder(*batch)
            embs.append(entity_emb.cpu())
        embs = torch.concat(embs, dim=0)
        t2i = {topic_id: idx for idx, topic_id in i2t.items()}
        return cls(t2i, embs)

    @classmethod
    def from_file(cls, fname, device):
        """
        Initialize the topic embeddings from saved file.
        :param fname: file location of the saved embeddings
        :param device: torch device to load the embeddings on
        :return: dataset containing topic embeddings
        """
        if device.type == "cpu":
            saved = torch.load(fname, map_location="cpu")
        else:
            saved = torch.load(fname)
        t2i, embs = saved["t2i"], saved["topic_embeddings"]

        return BiencTopicEmbeddings(t2i, embs)

    def to_file(self, fname):
        """Save topic embeddings to FNAME."""
        torch.save({"t2i": self.t2i,
                    "topic_embeddings": self.embs}, fname)
        print(f"Saved bienc topic embeddings to {fname}.")
