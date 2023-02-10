from pathlib import Path

import pandas as pd
import torch

from bienc.inference import embed_and_nn, entities_inference, predict_entities
from bienc.model import BiencoderModule
from config import NUM_NEIGHBORS
from data.content import get_content2text
from data.topics import get_topic2text
from typehints import FName


def get_test_topic_ids(fname: FName) -> list[str]:
    df = pd.read_csv(fname)
    return sorted(list(set(df["topic_id"])))


def get_biencoder(fname: FName, device: torch.device) -> BiencoderModule:
    model = BiencoderModule()
    model.to(device)
    if device.type == "cpu":
        checkpoint = torch.load(fname, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_submission_df(t2preds: dict[str, set[str]]) -> pd.DataFrame:
    def to_str(content_ids):
        return " ".join(sorted(list(content_ids)))
    topic_id_col = sorted(t2preds.keys())
    content_ids_col = [to_str(t2preds[topic_id]) for topic_id in topic_id_col]
    df = pd.DataFrame({"topic_id": topic_id_col, "content_ids": content_ids_col})
    return df



DATA_DIR = Path("../data")
BIENCODER_FNAME = "../out/0210-143245.pt"
THRESH = 0.18
batch_size = 128

device = torch.device("cuda")
encoder = get_biencoder(BIENCODER_FNAME, device)

content_df = pd.read_csv(DATA_DIR / "content.csv")
topics_df = pd.read_csv(DATA_DIR / "topics.csv")
topic_ids = get_test_topic_ids("../data/sample_submission.csv")
content_ids = sorted(list(set(content_df["id"])))
c2i = {content_id: content_idx for content_idx, content_id in enumerate(content_ids)}

topic2text = get_topic2text(topics_df)
content2text = get_content2text(content_df)

nn_model = embed_and_nn(encoder, content_ids, content2text, NUM_NEIGHBORS, batch_size, device)
distances, indices = entities_inference(topic_ids, encoder, nn_model, topic2text, device, batch_size)
t2preds = predict_entities(topic_ids, distances, indices, THRESH, c2i)
submission_df = get_submission_df(t2preds)
