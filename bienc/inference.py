# Inference for the Bi-encoder
import gc
from typing import Any, Dict, List, Set, Tuple

import cupy as cp
import pandas as pd
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


def entities_inference(data_ids: List[str], encoder: BiencoderModule, nn_model: NearestNeighbors,
                       data2text: Dict[str, str],
                       device: torch.device, batch_size: int) -> Tuple[Any, Any]:
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
    dset = BiencInferenceDataset(data_ids, data2text, CFG.CONTENT_NUM_TOKENS)
    loader = DataLoader(dset, batch_size=batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)
    embs = embed(encoder, loader, device)
    embs = cp.array(embs)
    distances, indices = nn_model.kneighbors(embs, return_distance=True)
    distances, indices = cp.asnumpy(distances), cp.asnumpy(indices)
    return distances, indices


def get_cand_df(topic_ids: List[str], distances, indices, e2i: Dict[str, int]) -> pd.DataFrame:
    """
    Converts distances and indices obtained from NN search to dataframe containing candidate contents for all topics.
    :param topic_ids: all topic ids in order
    :param distances: distances - output from NN model
    :param indices: indices - output from NN model
    :param e2i: dct mapping entity names to indices
    :return: dataframe with two columns: topic ids and concatenated candidate ids
    """
    i2e = {entity_idx: entity_id for entity_id, entity_idx in e2i.items()}
    all_topic_cand_ids = []
    for data_id, dists, idxs in zip(topic_ids, distances, indices):
        cands = [i2e[pred_idx] for pred_idx in idxs]
        # TODO
        # if not cands:
        #     cands = [i2e[idxs[0]]]
        all_topic_cand_ids.append(" ".join(cands))
    return pd.DataFrame({"topic_id": topic_ids, "cands": all_topic_cand_ids})
