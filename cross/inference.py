import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from cross.dset import CrossInferenceDataset


def predict(model, dset: CrossInferenceDataset, classifier_thresh: float, batch_size: int, device: torch.device):
    loader = DataLoader(dset, batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)
    all_preds = np.empty(len(dset), dtype=int)
    model.eval()
    i = 0
    for batch in tqdm(loader):
        batch = tuple(x.to(device) for x in batch)
        with torch.no_grad():
            logits = model(*batch)
        probs = logits.softmax(dim=1)[:, 1]
        bs = logits.shape[0]
        all_preds[i:(i+bs)] = (probs >= classifier_thresh).cpu().numpy()
        i += bs
    return all_preds
