import logging
import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)


def get_topic_embeddings(encoder, device, data_loader):
    """
    Get topic embeddings from a trained model.
    :param encoder: trained encoder of the bi-encoder
    :param device: device on which to compute embeddings
    :param data_loader: data loader of a BiencInferenceDataset over the topics
    :return: torch CPU tensor containing topic embeddings
    """
    embs = []

    logger.info("Preparing Bi-encoder inference dataset containing topic embeddings...")
    encoder.eval()
    encoder.to(device)

    for batch in tqdm(data_loader):
        batch = tuple(x.to(device) for x in batch)
        with torch.no_grad():
            emb = encoder(*batch)
        embs.append(emb.cpu())
    embs = torch.concat(embs, dim=0)
    return embs


def inference(encoder, loader, device):
    embs = []
    encoder.eval()
    for batch in tqdm(loader):
        batch = tuple(x.to(device) for x in batch)
        with torch.no_grad():
            emb = encoder(*batch)
            embs.append(emb.cpu())
    embs = torch.concat(embs, dim=0)
    return embs
