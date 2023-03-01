# Inference for the Bi-encoder
import gc
from typing import Dict, List, Set

import cupy as cp
import numpy as np
import torch
from cuml import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from bienc.dset import BiencInferenceDataset
from bienc.model import BiencoderModule
from config import CFG
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


def embed_data(encoder: BiencoderModule, data_ids: List[str], data2text: Dict[str, str],
               batch_size: int, device: torch.device):
    assert is_ordered(data_ids)
    dset = BiencInferenceDataset(data_ids, data2text, CFG.TOPIC_NUM_TOKENS)
    loader = DataLoader(dset, batch_size=batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)
    print("Preparing Bi-encoder inference dataset containing entity embeddings...")
    embs = embed(encoder, loader, device)
    return embs


def prepare_nn(embs, num_neighbors: int):
    embs = cp.array(embs)
    nn_model = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine')
    nn_model.fit(embs)
    return nn_model


def embed_and_nn(encoder: BiencoderModule, data_ids: List[str], data2text: Dict[str, str],
                 num_neighbors: int,
                 batch_size: int, device: torch.device):
    """Embeds and prepares nearest neighbors data structure."""
    embs = embed_data(encoder, data_ids, data2text, batch_size, device)
    # Rapids NN runs on GPU - shift model to CPU to save GPU memory
    encoder.to(torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
    nn_model = prepare_nn(embs, num_neighbors)
    # Move model back onto GPU. IMPORTANT: Need to tie optimizer to model parameters again before the next training loop.
    del embs
    gc.collect()
    torch.cuda.empty_cache()
    encoder.to(device)
    return nn_model


def predict_entities(topic_ids: List[str], distances, indices, thresh, e2i: Dict[str, int]) -> Dict[str, Set[str]]:
    i2e = {entity_idx: entity_id for entity_id, entity_idx in e2i.items()}
    t2preds = {}
    for data_id, dists, idxs in zip(topic_ids, distances, indices):
        t2preds[data_id] = set(i2e[idx] for idx in idxs[dists <= thresh])
        if not t2preds[data_id]:
            t2preds[data_id] = set([i2e[idxs[0]]])
    return t2preds


def entities_inference(topic_ids: List[str], encoder: BiencoderModule, nn_model: NearestNeighbors,
                       t2text: Dict[str, str],
                       device: torch.device, batch_size: int):
    """
    Embed data and find their nearest neighbors among entities.
    :param topic_ids: topic ids for which we are performing inference
    :param encoder: trained Bi-encoder encoder
    :param nn_model: Rapids nearest neighbor data structure
    :param t2text: dict mapping topic ids to their text representations
    :param device: device we are running inference on
    :param batch_size: batch size to use
    :return: indices: numpy array of shape (num_examples, num_neighbors)
    """
    dset = BiencInferenceDataset(topic_ids, t2text, CFG.CONTENT_NUM_TOKENS)
    loader = DataLoader(dset, batch_size=batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)
    embs = embed(encoder, loader, device)
    embs = cp.array(embs)
    indices = nn_model.kneighbors(embs, return_distance=False)
    indices = cp.asnumpy(indices)
    return indices


def filter_languages(indices, topic_ids: List[str], c2i: Dict[str, int], t2lang: Dict[str, str], c2lang: Dict[str, str]):
    """
    Filters predicted indices by only allowing candidates that match the topic language.
    Indices for non-matching languages are deleted and their successors move up in rank.
    Towards the end of the array where don't end up with any predictions -1 fills up the empty spots in the indices array.
    :param indices: numpy array output by entities_inference of shape (num_topics, num_neighbors)
    :param topic_ids: topic ids in the right order of inference
    :param c2i: dict mapping content ids to content indices
    :param t2lang: dict mapping topic ids to topic languages
    :param c2lang: dict mapping content ids to content languages
    :return: modified indices numpy array of shape (num_topics, num_neighbors)
    """
    i2c = {content_idx: content_id for content_id, content_idx in c2i.items()}
    for i, (topic_id, idxs) in enumerate(zip(topic_ids, indices)):
        topic_lang = t2lang[topic_id]
        matching_lang_idxs = [idx for idx in idxs if c2lang[i2c[idx]] == topic_lang]
        if matching_lang_idxs:
            indices[i, :len(matching_lang_idxs)] = np.array(matching_lang_idxs, dtype=float)
            indices[i, len(matching_lang_idxs):] = -1
        else:
            indices[i, :] = -1

    return indices


def mistery(encoder: BiencoderModule,
            topic_ids: List[str], content_ids: List[str],
            topic2text: Dict[str, str], content2text: Dict[str, str],
            filter_lang: bool, t2lang: Dict[str, str], c2lang: Dict[str, str], c2i: Dict[str, int],
            batch_size: int, device: torch.device):
    # Prepare nearest neighbors data structure for entities
    nn_model = embed_and_nn(encoder, content_ids, content2text, CFG.NUM_NEIGHBORS, batch_size, device)

    # Get nearest neighbor distances and indices
    indices = entities_inference(topic_ids, encoder, nn_model, topic2text, device, batch_size)

    # Filter languages
    if filter_lang:
        c2i = c2i.copy()
        c2i["dummy"] = -1
        indices = filter_languages(indices, topic_ids, c2i, t2lang, c2lang)

    return indices
