import cupy as cp
from pathlib import Path
import pandas as pd
import torch
from cuml import NearestNeighbors
from torch.utils.data import DataLoader

from bienc.dset import BiencInferenceDataset
from bienc.inference import embed, inference
from bienc.model import BiencoderModule
from config import NUM_NEIGHBORS, TOPIC_NUM_TOKENS, NUM_WORKERS, CONTENT_NUM_TOKENS
from data.content import get_content2text
from data.topics import get_topic2text
from utils import flatten_content_ids


def get_test_topic_ids(fname):
    df = pd.read_csv(fname)
    return sorted(list(set(df["topic_id"])))


def get_biencoder(fname, device):
    model = BiencoderModule()
    model.to(device)
    if device.type == "cpu":
        checkpoint = torch.load(fname, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


DATA_DIR = Path("../data")
batch_size = 128

device = torch.device("cpu")
topic_ids = get_test_topic_ids("../data/sample_submission.csv")
encoder = get_biencoder("../out/0208-163213.pt", device)

content_df = pd.read_csv(DATA_DIR / "content.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv")

topic2text = get_topic2text(topics_df)
content2text = get_content2text(content_df)

topic_dset = BiencInferenceDataset(list(topic_ids), topic2text, TOPIC_NUM_TOKENS)
topic_loader = DataLoader(topic_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
topic_embs = embed(encoder, topic_loader, device)
topic_embs = cp.array(topic_embs)
nn_model = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, metric='cosine')
nn_model.fit(topic_embs)

content_ids = sorted(list(set(content_df["id"])))
content_dset = BiencInferenceDataset(content_ids, content2text, CONTENT_NUM_TOKENS)
content_loader = DataLoader(content_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
content_embs = inference(encoder, content_loader, device)
content_embs_gpu = cp.array(content_embs)
indices = nn_model.kneighbors(content_embs_gpu, return_distance=False)
indices = cp.asnumpy(indices)

c2gold = get_content_id_gold(gold_df)
ranks = get_ranks(indices, content_ids, c2gold, t2i)
mir = get_mean_inverse_rank(ranks)
recall_dct = get_recall_dct(ranks)