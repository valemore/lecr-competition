# Inference for the Bi-encoder
from typing import Any

import cupy as cp
import torch
from cuml import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from bienc.dset import BiencInferenceDataset
from bienc.model import BiencoderModule
from config import NUM_WORKERS, TOPIC_NUM_TOKENS, NUM_NEIGHBORS, CONTENT_NUM_TOKENS
from utils import is_ordered


def embed(encoder: BiencoderModule, data_loader: DataLoader, device: torch.device):
    """
    Get embeddings from a trained model.
    :param encoder: trained encoder of the bi-encoder
    :param device: device on which to compute embeddings
    :param data_loader: data loader of a BiencInferenceDataset over topics or contents
    :return: torch CPU tensor containing topic embeddings
    """
    embs = []
    encoder.eval()
    encoder.to(device)
    for batch in tqdm(data_loader):
        batch = tuple(x.to(device) for x in batch)
        with torch.no_grad():
            emb = encoder(*batch)
        embs.append(emb.cpu())
    embs = torch.concat(embs, dim=0)
    return embs


def embed_topics_nn(encoder: BiencoderModule, topic_ids: list[str], topic2text: dict[str, str],
                    num_neighbors: int,
                    batch_size: int, device: torch.device):
    """Embeds topics and prepares nearest neighbors data structure."""
    assert is_ordered(topic_ids)
    topic_dset = BiencInferenceDataset(topic_ids, topic2text, TOPIC_NUM_TOKENS)
    topic_loader = DataLoader(topic_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    print("Preparing Bi-encoder inference dataset containing topic embeddings...")
    topic_embs = embed(encoder, topic_loader, device)
    topic_embs = cp.array(topic_embs)
    nn_model = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
    nn_model.fit(topic_embs)
    return nn_model


def predict_topics(content_ids: list[str], distances, indices, thresh, t2i: dict[str, int]) -> dict[str, list[str]]:
    i2t = {topic_idx: topic_id for topic_id, topic_idx in t2i.items()}
    c2preds = {}
    for content_id, dists, idxs in zip(content_ids, distances, indices):
        c2preds[content_id] = [i2t[idx] for idx in idxs[dists <= thresh].tolist()]
    return c2preds


def bienc_inference(content_ids: list[str], encoder: BiencoderModule, nn_model: NearestNeighbors,
                    content2text: dict[str, str],
                    device: torch.device, batch_size: int) -> tuple[Any, Any]:
    """
    Embed contents and find their nearest neighbors among topics.
    :param content_ids: content ids for which we are performing inference
    :param encoder: trained Bi-encoder encoder
    :param nn_model: Rapids nearest neighbor data structure
    :param content2text: dict mapping content ids to their text representations
    :param device: device we are running inference on
    :param batch_size: batch size to use
    :return: distances, indices: both are numpy arrays of shape (num_examples, num_neighbors)
    """
    content_dset = BiencInferenceDataset(content_ids, content2text, CONTENT_NUM_TOKENS)
    content_loader = DataLoader(content_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    content_embs = embed(encoder, content_loader, device)
    content_embs_gpu = cp.array(content_embs)
    distances, indices = nn_model.kneighbors(content_embs_gpu, return_distance=True)
    distances, indices = cp.asnumpy(distances), cp.asnumpy(indices)
    return distances, indices