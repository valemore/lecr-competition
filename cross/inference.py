import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from cross.dset import CrossInferenceDataset


def predict(model, dset: CrossInferenceDataset, classifier_thresh: float, batch_size: int, device: torch.device):
    loader = DataLoader(dset, batch_size, num_workers=CFG.NUM_WORKERS, shuffle=False)
    all_preds = []
    model.eval()
    for batch in tqdm(loader):
        batch = tuple(x.to(device) for x in batch)
        with torch.no_grad():
            out = model(*batch)
        logits = out.logits
        probs = logits.softmax(dim=1)[:, 1]
        all_preds.append((probs >= classifier_thresh).cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0).astype(int)
    return all_preds
