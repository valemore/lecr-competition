from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import torch

from bienc.inference import embed_and_nn, entities_inference, predict_entities
from bienc.model import BiencoderModule
from bienc.tokenizer import init_tokenizer
from config import NUM_NEIGHBORS
from data.content import get_content2text
from data.topics import get_topic2text
from typehints import FName


def get_test_topic_ids(fname: FName) -> List[str]:
    df = pd.read_csv(fname)
    return sorted(list(set(df["topic_id"])))


def get_biencoder(biencoder_dir: FName, device: torch.device) -> BiencoderModule:
    model = BiencoderModule(biencoder_dir)
    model.to(device)
    model.eval()
    return model


def get_submission_df(t2preds: Dict[str, Set[str]]) -> pd.DataFrame:
    def to_str(content_ids):
        return " ".join(sorted(list(content_ids)))
    topic_id_col = sorted(t2preds.keys())
    content_ids_col = [to_str(t2preds[topic_id]) for topic_id in topic_id_col]
    df = pd.DataFrame({"topic_id": topic_id_col, "content_ids": content_ids_col})
    return df

# DATA_DIR = Path("/kaggle/input/learning-equality-curriculum-recommendations")
# BIENCODER_FNAME = "/kaggle/input/kolibri-model/biencoder.pt"

def main(data_dir: FName, tokenizer_dir: FName, biencoder_dir: FName, batch_size: int):
    THRESH = 0.18

    data_dir = Path(data_dir)
    device = torch.device("cuda")
    init_tokenizer(tokenizer_dir)
    encoder = get_biencoder(biencoder_dir, device)

    content_df = pd.read_csv(data_dir / "content.csv")
    topics_df = pd.read_csv(data_dir / "topics.csv")
    topic_ids = get_test_topic_ids("../data/sample_submission.csv")
    content_ids = sorted(list(set(content_df["id"])))
    c2i = {content_id: content_idx for content_idx, content_id in enumerate(content_ids)}

    topic2text = get_topic2text(topics_df)
    content2text = get_content2text(content_df)

    nn_model = embed_and_nn(encoder, content_ids, content2text, NUM_NEIGHBORS, batch_size, device)
    distances, indices = entities_inference(topic_ids, encoder, nn_model, topic2text, device, batch_size)
    t2preds = predict_entities(topic_ids, distances, indices, THRESH, c2i)
    submission_df = get_submission_df(t2preds)
    return submission_df
