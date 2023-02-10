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


def embed_data(encoder: BiencoderModule, data_ids: list[str], data2text: dict[str, str],
               batch_size: int, device: torch.device):
    assert is_ordered(data_ids)
    dset = BiencInferenceDataset(data_ids, data2text, TOPIC_NUM_TOKENS)
    loader = DataLoader(dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    print("Preparing Bi-encoder inference dataset containing entity embeddings...")
    embs = embed(encoder, loader, device)
    return embs


def prepare_nn(embs, num_neighbors: int):
    embs = cp.array(embs)
    nn_model = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
    nn_model.fit(embs)
    return nn_model


def embed_and_nn(encoder: BiencoderModule, data_ids: list[str], data2text: dict[str, str],
                 num_neighbors: int,
                 batch_size: int, device: torch.device):
    """Embeds and prepares nearest neighbors data structure."""
    embs = embed_data(encoder, data_ids, data2text, batch_size, device)
    nn_model = prepare_nn(embs, num_neighbors)
    return nn_model


def predict_entities(topic_ids: list[str], distances, indices, thresh, e2i: dict[str, int]) -> dict[str, set[str]]:
    i2e = {entity_idx: entity_id for entity_id, entity_idx in e2i.items()}
    t2preds = {}
    for data_id, dists, idxs in zip(topic_ids, distances, indices):
        t2preds[data_id] = set(i2e[idx] for idx in idxs[dists <= thresh])
        if not t2preds[data_id]:
            t2preds[data_id] = set(i2e[idxs[0]])
    return t2preds


def entities_inference(data_ids: list[str], encoder: BiencoderModule, nn_model: NearestNeighbors,
                       data2text: dict[str, str],
                       device: torch.device, batch_size: int) -> tuple[Any, Any]:
    """
    Embed data and find their nearest neighbors among entities.
    :param data_ids: data ids for which we are performing inference
    :param encoder: trained Bi-encoder encoder
    :param nn_model: Rapids nearest neighbor data structure
    :param data2text: dict mapping data ids to their text representations
    :param device: device we are running inference on
    :param batch_size: batch size to use
    :return: distances, indices: both are numpy arrays of shape (num_examples, num_neighbors)
    """
    dset = BiencInferenceDataset(data_ids, data2text, CONTENT_NUM_TOKENS)
    loader = DataLoader(dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    embs = embed(encoder, loader, device)
    embs = cp.array(embs)
    distances, indices = nn_model.kneighbors(embs, return_distance=True)
    distances, indices = cp.asnumpy(distances), cp.asnumpy(indices)
    return distances, indices
