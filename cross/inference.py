import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from cross.dset import CrossInferenceDataset


def predict_probs(model, dset: CrossInferenceDataset, batch_size: int, device: torch.device):
    loader = DataLoader(dset, batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)
    all_probs = np.empty(len(dset), dtype=int)
    model.eval()
    i = 0
    for batch in tqdm(loader):
        batch = tuple(x.to(device) for x in batch)
        with torch.no_grad():
            logits = model(*batch)
        probs = logits.softmax(dim=1)[:, 1]
        bs = logits.shape[0]
        all_probs[i:(i+bs)] = probs.cpu().numpy()
        i += bs
    return all_probs
